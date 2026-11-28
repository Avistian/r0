"""Strict pair persistence for interventions with immutable baseline target calibration."""

from dataclasses import asdict
from typing import Any

from r0.modules.priors.lib.calibrated_population_intervention import (
    CalibratedPopulationIntervention,
    intervene_calibrated_population,
)
from r0.modules.priors.lib.calibrated_population_intervention_output import (
    generate_calibrated_intervention,
)
from r0.modules.priors.lib.causal_intervention_records import CausalInterventionRecipe
from r0.modules.priors.lib.causal_limits import CausalLimits
from r0.modules.priors.lib.population_calibrated_output import generate_calibrated_population
from r0.modules.priors.lib.population_graph_comparison import population_graph_pair_card
from r0.modules.priors.lib.population_target_calibration import CalibratedPopulationGraph
from r0.modules.priors.lib.population_task_semantics import (
    population_task_record,
    validate_population_task,
)
from r0.modules.priors.lib.representatives import encode
from r0.modules.priors.lib.training_task import TrainingTask


def calibrated_pair_card(
    base: CalibratedPopulationGraph,
    changed: CalibratedPopulationIntervention,
    before: TrainingTask,
    after: TrainingTask,
) -> dict[str, Any]:
    return {
        **population_graph_pair_card(base.original, changed.original, before, after),
        "version": "calibrated-population-pair-v1",
        "calibration_equal": True,
        "calibration_identity": {
            "seed": base.calibration_population.seed,
            "task_id": base.calibration_population.task_id,
        },
        "calibration_digest": base.calibration.sample_digest,
        "variance_semantics": (
            "fixed baseline independent calibration; changed variance is not matched"
        ),
        "query_fit": False,
    }


def generate_calibrated_population_pair(
    base: CalibratedPopulationGraph, change: CausalInterventionRecipe
):
    from r0.modules.priors.lib.pairs import TaskPair

    changed = intervene_calibrated_population(base, change)
    before = generate_calibrated_population(
        base, seed=base.original.seed, task_id=base.original.task_id
    )
    after = generate_calibrated_intervention(
        changed, seed=base.original.seed, task_id=base.original.task_id
    )
    return TaskPair(before, after, encode(calibrated_pair_card(base, changed, before, after)))


def compare_calibrated_population_pair(
    before: TrainingTask, after: TrainingTask, *, limits: CausalLimits | None
):
    from r0.modules.priors.lib.generate import generate_task, recipe_from_dict

    b, a = population_task_record(before), population_task_record(after)
    raw = a["recipe"]
    if (
        raw.get("version") != "calibrated-population-intervention-v1"
        or encode(raw.get("base")) != encode(b["recipe"])
        or (b["seed"], b["task_id"]) != (a["seed"], a["task_id"])
    ):
        raise ValueError("calibrated comparison requires the exact baseline and row identities")
    if any(r["execution"]["implementation"] != r["recipe"].get("version") for r in (b, a)):
        raise ValueError("calibrated comparison implementation identity differs")
    recipe = recipe_from_dict(raw, limits=limits)
    if not isinstance(recipe, CalibratedPopulationIntervention):
        raise ValueError("calibrated comparison requires its complete intervention archive")
    expected_before = generate_task(recipe.base, seed=b["seed"], task_id=b["task_id"])
    expected_after = generate_task(recipe, seed=a["seed"], task_id=a["task_id"])
    validate_population_task(before, expected_before, b)
    validate_population_task(after, expected_after, a)
    if encode(asdict(recipe.base)) != encode(b["recipe"]):
        raise ValueError("calibrated baseline construction differs")
    return calibrated_pair_card(recipe.base, recipe, before, after)
