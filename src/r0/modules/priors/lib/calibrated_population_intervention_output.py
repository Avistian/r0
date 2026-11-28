"""Apply baseline calibration after changed structural execution and before its target adapter."""

import json
from dataclasses import asdict, replace

from r0.modules.priors.lib.calibrated_population_intervention import (
    CalibratedPopulationIntervention,
)
from r0.modules.priors.lib.causal_intervention_execution import InterventionExecution
from r0.modules.priors.lib.functional_causal import FunctionalCausalRecipe
from r0.modules.priors.lib.functional_causal_evaluation import functional_rng
from r0.modules.priors.lib.population_calibrated_output import apply_population_calibration
from r0.modules.priors.lib.population_generation_identity import preflight_calibrated_generation
from r0.modules.priors.lib.recursive_population_graph_intervention import (
    generate_population_graph_intervention,
)
from r0.modules.priors.lib.representatives import encode, record_fingerprint
from r0.modules.priors.lib.training_task import TrainingTask


def generate_calibrated_intervention(
    recipe: CalibratedPopulationIntervention, *, seed: int, task_id: int, device: str = "cpu"
) -> TrainingTask:
    preflight_calibrated_generation(recipe, seed=seed, task_id=task_id, device=device)
    recipe.validate()
    raw = generate_population_graph_intervention(
        recipe.original, seed=seed, task_id=task_id, device=device
    )
    execution = InterventionExecution(recipe.original.change)
    scores, targets, probabilities, classes = apply_population_calibration(
        raw,
        recipe.calibration,
        adapter=recipe.task
        if isinstance(recipe.base.original.request.base, FunctionalCausalRecipe)
        else None,
        rng=lambda key: execution.rng(lambda *keys: functional_rng(seed, task_id, *keys), key),
    )
    config = asdict(recipe)
    record = json.loads(raw.recipe_json)
    record["recipe"] = config
    record["execution"]["implementation"] = recipe.version
    source = record_fingerprint((config, seed, task_id))
    return replace(
        raw,
        source_task_id=source,
        episode_id=record_fingerprint((source, recipe.task.split, recipe.task.context_rows)),
        scores=scores,
        targets=targets,
        probabilities=probabilities,
        classes=classes,
        recipe_json=encode(record),
        mechanism_json=encode(
            {
                "family": "calibrated_population_graph",
                "version": recipe.version,
                "original": json.loads(raw.mechanism_json),
                "calibration": asdict(recipe.calibration),
                "calibration_identity": {
                    "seed": recipe.base.calibration_population.seed,
                    "task_id": recipe.base.calibration_population.task_id,
                },
                "calibration_lifecycle": "fixed-baseline-independent-population",
                "query_fit": False,
            }
        ),
    )
