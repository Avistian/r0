"""Validated before/after GP-root interventions with explicit population semantics."""

from typing import Any

import torch

from r0.modules.priors.lib.causal_intervention_records import CausalInterventionRecipe
from r0.modules.priors.lib.causal_limits import CausalLimits
from r0.modules.priors.lib.gp_column_decoding import gp_payload_equal
from r0.modules.priors.lib.gp_root_intervention import (
    GPRootInterventionSource,
    intervene_gp_root_dag,
)
from r0.modules.priors.lib.gp_root_intervention_decoding import preflight_gp_intervention
from r0.modules.priors.lib.gp_root_intervention_inspection import validate_gp_intervention_task
from r0.modules.priors.lib.gp_root_intervention_output import generate_gp_root_intervention
from r0.modules.priors.lib.gp_root_task_validation import gp_root_task_record, validate_gp_root_task
from r0.modules.priors.lib.representatives import encode, record_fingerprint
from r0.modules.priors.lib.training_task import TrainingTask


def generate_gp_root_pair(
    base: GPRootInterventionSource,
    change: CausalInterventionRecipe,
    *,
    limits: CausalLimits | None = None,
):
    from r0.modules.priors.lib.generate import generate_task
    from r0.modules.priors.lib.pairs import TaskPair

    after_recipe = intervene_gp_root_dag(base, change, limits=limits)
    seed, task_id = after_recipe.original.source.seed, after_recipe.original.source.task_id
    before = generate_task(base, seed=seed, task_id=task_id)
    after = generate_gp_root_intervention(after_recipe, seed=seed, task_id=task_id)
    return TaskPair(
        before, after, encode(compare_gp_root_pair(before, after, limits=after_recipe.limits))
    )


def compare_gp_root_pair(
    before: TrainingTask, after: TrainingTask, *, limits: CausalLimits | None = None
) -> dict[str, Any]:
    cap = limits or CausalLimits()
    b, a = gp_root_task_record(before), gp_root_task_record(after)
    raw, _ = preflight_gp_intervention(a["recipe"], limits=cap)
    if not gp_payload_equal(b["recipe"], raw["base"]) or any(
        type(record[key]) is not int or record[key] != a[key]
        for record in (b, a)
        for key in ("seed", "task_id")
    ):
        raise ValueError("GP intervention comparison must retain its complete original source")
    validate_gp_root_task(before, limits=cap)
    recipe, _ = validate_gp_intervention_task(after, limits=cap)
    equalities = {
        "context_indices": torch.equal(before.context_indices, after.context_indices),
        "query_indices": torch.equal(before.query_indices, after.query_indices),
        "classes": before.classes == after.classes,
        "complete_source": True,
        "fixed_independent_fit": True,
    }
    if not all(equalities.values()):
        raise ValueError("GP intervention comparison changed fixed row/split/type contracts")
    return {
        "version": "gp-root-intervention-pair-v1",
        "kind": "causal-intervention",
        "pair_id": record_fingerprint(
            ("gp-root-intervention-pair-v1", before.episode_id, after.episode_id)
        ),
        "mode": recipe.change.mode,
        "population": recipe.change.population,
        "seed": a["seed"],
        "task_id": a["task_id"],
        "original_population_id": recipe.original.source.population_id,
        "intervention_population_id": recipe.population.population_id,
        "observed_equalities": equalities,
        "changed_target_fraction": float(
            (before.targets != after.targets).to(torch.float64).mean()
        ),
        "target_mean_before": float(before.targets.to(torch.float64).mean()),
        "target_mean_after": float(after.targets.to(torch.float64).mean()),
        "query_fit": False,
        "population_semantics": (
            "complete original GP population and unchanged keyed structural draws"
            if recipe.change.mode == "matched"
            else "same GP law; separately keyed complete population and fresh structural draws"
        ),
        "calibration_semantics": (
            "retain complete original independent target fit; never fit after intervention"
        ),
    }
