"""Paired target edits on one explicit GP environment, with shared adapter streams."""

import json
from dataclasses import asdict
from typing import Any

import torch

from r0.modules.priors.lib.causal_limits import CausalLimits
from r0.modules.priors.lib.function_metadata import admit_function_metadata
from r0.modules.priors.lib.gp_column_comparison import compare_gp_columns, gp_comparison_controls
from r0.modules.priors.lib.gp_column_decoding import (
    gp_fields,
    gp_payload_equal,
    gp_population_metadata,
)
from r0.modules.priors.lib.gp_column_metadata import gp_sequence, gp_target_metadata
from r0.modules.priors.lib.gp_column_output import GPColumnRecipe
from r0.modules.priors.lib.gp_column_population import GPColumnPopulation
from r0.modules.priors.lib.gp_column_target import GPColumnTarget
from r0.modules.priors.lib.gp_target_environment import (
    GPColumnTargetRecipe,
    admit_gp_target_environment,
    generate_gp_target_environment,
    gp_column_target_recipe_from_dict,
    gp_target_environment_metadata,
    gp_target_record_metadata,
)
from r0.modules.priors.lib.gp_task_validation import validate_gp_task
from r0.modules.priors.lib.linear import LinearRecipe
from r0.modules.priors.lib.representatives import encode, record_fingerprint
from r0.modules.priors.lib.training_task import TrainingTask

_DEFAULT_LIMITS = CausalLimits()


def generate_gp_column_pair(
    original: GPColumnRecipe,
    replacement: GPColumnTarget,
    *,
    mode: str = "matched",
    fresh_seed: int | None = None,
    limits: CausalLimits = _DEFAULT_LIMITS,
):
    from r0.modules.priors.lib.pairs import TaskPair

    if type(original) is not GPColumnRecipe or type(original.source) is not GPColumnPopulation:
        raise ValueError("GP target edits require an original realized population")
    if type(replacement) is not GPColumnTarget:
        raise ValueError("GP target edits require an exact frozen target")
    admit_gp_target_environment(original.request, replacement, original.task, limits)
    comparison = compare_gp_columns(
        original.source, replacement, mode=mode, fresh_seed=fresh_seed, limits=limits
    )
    before, after = (
        generate_gp_target_environment(
            GPColumnTargetRecipe(comparison, original.task, side),
            seed=original.source.seed,
            task_id=original.source.task_id,
        )
        for side in ("original", "replacement")
    )
    return TaskPair(before, after, encode(compare_gp_column_pair(before, after, limits=limits)))


def compare_gp_column_pair(
    before: TrainingTask, after: TrainingTask, *, limits: CausalLimits = _DEFAULT_LIMITS
) -> dict[str, Any]:
    records = [json.loads(t.recipe_json) for t in (before, after)]
    for record, side in zip(records, ("original", "replacement"), strict=True):
        gp_target_record_metadata(record, limits=limits)
        raw, _ = gp_target_environment_metadata(record["recipe"], limits=limits)
        p = raw["comparison"]["original"]
        if raw["side"] != side or any(
            type(record.get(k)) is not int or record[k] != p[k] for k in ("seed", "task_id")
        ):
            raise ValueError("GP pair side or exact original identity differs")
    b, a = [r["recipe"] for r in records]
    if not gp_payload_equal(b, {**a, "side": "original"}):
        raise ValueError("GP pair must retain one complete comparison and adapter")
    recipes = [gp_column_target_recipe_from_dict(r, limits=limits) for r in (b, a)]
    for actual, recipe, record in zip((before, after), recipes, records, strict=True):
        validate_gp_task(
            actual,
            generate_gp_target_environment(recipe, seed=record["seed"], task_id=record["task_id"]),
            recipe.version,
        )
    c = recipes[0].comparison
    equalities = {
        "inputs": torch.equal(before.inputs, after.inputs),
        "context_indices": torch.equal(before.context_indices, after.context_indices),
        "query_indices": torch.equal(before.query_indices, after.query_indices),
        "classes": before.classes == after.classes,
        "original_population": True,
        "adapter_streams": True,
        "calibration_inputs": True,
    }
    if not all(equalities.values()):
        raise ValueError("GP target pair violated shared environment invariants")
    return {
        "version": "gp-target-pair-v1",
        "kind": "gp-target-edit",
        "pair_id": record_fingerprint(("gp-target-pair-v1", before.episode_id, after.episode_id)),
        "seed": c.original.seed,
        "task_id": c.original.task_id,
        "original_population_id": c.original.population_id,
        "mode": c.mode,
        "fresh_seed": c.fresh_seed,
        "population": "original locations/kernels; shared innovations on both sides",
        "before": {
            "episode_id": before.episode_id,
            "target": json.loads(encode(asdict(c.original.target))),
        },
        "after": {
            "episode_id": after.episode_id,
            "target": json.loads(encode(asdict(c.replacement))),
        },
        "required_equalities": equalities,
        "observed_equalities": equalities,
        "changed_target_fraction": float(
            (before.targets != after.targets).to(torch.float64).mean()
        ),
        "query_fit": False,
        "calibration": "each target uses the same original independent marginal calibration inputs",
        "notes": [
            "Fresh mode evaluates both targets on fresh innovations at fixed original locations.",
            "Target edits are structural equation comparisons, not conditional GP inference.",
            "Comparison identity is fixed; fresh seed is separate from original population ID.",
        ],
    }


def gp_column_target_from_dict(
    data: Any, *, limits: CausalLimits = _DEFAULT_LIMITS
) -> GPColumnTarget:
    admit_function_metadata(data, limits)
    return gp_target_metadata(data)


def validate_gp_target_edit_controls(
    original: Any,
    replacement: Any,
    *,
    mode: str = "matched",
    fresh_seed: int | None = None,
    limits: CausalLimits = _DEFAULT_LIMITS,
) -> None:
    """Metadata-only control/aggregate validation, before original population replay."""
    admit_function_metadata(original, limits)
    raw = gp_fields(original, GPColumnRecipe)
    task_raw = gp_fields(raw["task"], LinearRecipe)
    gp_sequence(task_raw["class_offsets"])
    task_raw["class_offsets"] = tuple(task_raw["class_offsets"])
    task = LinearRecipe(**task_raw)
    request = gp_population_metadata(raw["source"], limits=limits)
    target = gp_column_target_from_dict(replacement, limits=limits)
    gp_comparison_controls(request, target, mode, fresh_seed)
    admit_gp_target_environment(request, target, task, limits)
