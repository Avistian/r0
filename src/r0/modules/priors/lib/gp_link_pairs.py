"""Linked target pairs retain the exact parent task and fixed comparison streams."""

import json
from dataclasses import asdict
from typing import Any, cast

import torch

from r0.modules.priors.lib.causal_limits import CausalLimits
from r0.modules.priors.lib.gp_column_decoding import gp_payload_equal
from r0.modules.priors.lib.gp_column_target import GPColumnTarget
from r0.modules.priors.lib.gp_target_environment import (
    GPColumnTargetRecipe,
    gp_target_record_metadata,
)
from r0.modules.priors.lib.gp_target_links import (
    GPColumnTargetLink,
    fit_gp_target_link,
    gp_column_target_link_from_dict,
    gp_link_admission,
    gp_target_base,
    linked_targets,
)
from r0.modules.priors.lib.gp_task_validation import validate_gp_task
from r0.modules.priors.lib.joint_function_admission import function_record_cost
from r0.modules.priors.lib.representatives import encode, record_fingerprint
from r0.modules.priors.lib.training_task import TrainingTask

_DEFAULT_LIMITS = CausalLimits()


def generate_gp_link_pair(
    parent: GPColumnTargetRecipe | GPColumnTargetLink,
    target: GPColumnTarget,
    *,
    limits: CausalLimits = _DEFAULT_LIMITS,
):
    from r0.modules.priors.lib.generate import generate_task, recipe_from_dict
    from r0.modules.priors.lib.pairs import TaskPair

    base = gp_target_base(parent)
    gp_link_admission(base, (*linked_targets(parent), target), limits)
    function_record_cost(parent, limits)
    retained = recipe_from_dict(asdict(parent), limits=limits)
    if not isinstance(retained, (GPColumnTargetRecipe, GPColumnTargetLink)):
        raise ValueError("GP linked target requires a retained environment parent")
    child = fit_gp_target_link(retained, target, limits=limits)
    p = base.comparison.original
    before = generate_task(retained, seed=p.seed, task_id=p.task_id)
    after = generate_task(child, seed=p.seed, task_id=p.task_id)
    return TaskPair(before, after, encode(compare_gp_link_pair(before, after, limits=limits)))


def compare_gp_link_pair(
    before: TrainingTask, after: TrainingTask, *, limits: CausalLimits = _DEFAULT_LIMITS
) -> dict[str, Any]:
    from r0.modules.priors.lib.generate import generate_task

    records = [json.loads(t.recipe_json) for t in (before, after)]
    for record in records:
        gp_target_record_metadata(record, limits=limits)
    b, a = records
    if a["recipe"].get("version") != "gp-column-target-link-v1" or not gp_payload_equal(
        a["recipe"]["parent"], b["recipe"]
    ):
        raise ValueError("GP linked pair must retain the exact parent environment")
    if any(a[k] != b[k] for k in ("seed", "task_id")):
        raise ValueError("GP linked pair must retain its original population identity")
    recipe = gp_column_target_link_from_dict(a["recipe"], limits=limits)
    for actual, selected in ((before, recipe.parent), (after, recipe)):
        expected = generate_task(selected, seed=a["seed"], task_id=a["task_id"])
        validate_gp_task(actual, expected, selected.version)
    base = gp_target_base(recipe)
    c = base.comparison
    parent = recipe.parent
    before_target = (
        parent.target
        if isinstance(parent, GPColumnTargetLink)
        else (c.replacement if parent.side == "replacement" else c.original.target)
    )
    same = {
        "inputs": torch.equal(before.inputs, after.inputs),
        "context_indices": torch.equal(before.context_indices, after.context_indices),
        "query_indices": torch.equal(before.query_indices, after.query_indices),
        "classes": before.classes == after.classes,
        "original_population": True,
        "adapter_streams": True,
        "calibration_inputs": True,
        "exact_parent_environment": True,
    }
    if not all(same.values()):
        raise ValueError("GP linked pair violated fixed environment invariants")
    return {
        "version": "gp-target-link-pair-v1",
        "kind": "gp-target-edit",
        "pair_id": record_fingerprint(
            ("gp-target-link-pair-v1", before.episode_id, after.episode_id)
        ),
        "seed": a["seed"],
        "task_id": a["task_id"],
        "mode": c.mode,
        "fresh_seed": c.fresh_seed,
        "original_population_id": c.original.population_id,
        "parent_environment_fingerprint": record_fingerprint(asdict(parent)),
        "lineage_depth": len(linked_targets(recipe)),
        "before": {
            "episode_id": before.episode_id,
            "target": json.loads(encode(asdict(before_target))),
        },
        "after": {
            "episode_id": after.episode_id,
            "target": json.loads(encode(asdict(recipe.target))),
        },
        "required_equalities": same,
        "observed_equalities": same,
        "changed_target_fraction": float(
            (before.targets != after.targets).to(torch.float64).mean()
        ),
        "query_fit": False,
        "notes": [
            "Before is the exact parent selected environment.",
            "Both sides retain the same fixed feature and adapter streams.",
            "Only the new target is calibrated on original independent marginal inputs.",
        ],
    }


def validate_gp_link_controls(
    parent: Any, replacement: Any, *, limits: CausalLimits = _DEFAULT_LIMITS
) -> None:
    """Admit all existing lineage and the next target before numerical replay."""
    from r0.modules.priors.lib.function_metadata import admit_function_metadata
    from r0.modules.priors.lib.gp_column_decoding import gp_population_metadata
    from r0.modules.priors.lib.gp_column_metadata import gp_target_metadata
    from r0.modules.priors.lib.gp_target_environment import (
        admit_gp_target_environment,
        gp_target_environment_metadata,
    )
    from r0.modules.priors.lib.gp_target_links import gp_link_extra, gp_target_link_metadata

    admit_function_metadata(parent, limits)
    admit_function_metadata(replacement, limits)
    if (
        type(parent) is dict
        and cast(dict[str, Any], parent).get("version") == "gp-column-target-link-v1"
    ):
        base, _, targets = gp_target_link_metadata(parent, limits=limits)
        _, task = gp_target_environment_metadata(base, limits=limits)
    else:
        base, task = gp_target_environment_metadata(parent, limits=limits)
        targets = ()
    if len(targets) >= 8:
        raise ValueError("GP target lineage exceeds explicit eight-link limit")
    request = gp_population_metadata(base["comparison"]["original"], limits=limits)
    extra = gp_link_extra(request, (*targets, gp_target_metadata(replacement)), limits)
    admit_gp_target_environment(
        request,
        gp_target_metadata(base["comparison"]["replacement"]),
        task,
        limits,
        extra_numbers=extra,
    )
