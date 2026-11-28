"""Apply fixed independent target scaling while preserving inputs and split."""

import json
from collections.abc import Callable
from dataclasses import asdict, replace

import numpy as np
import torch

from r0.modules.priors.lib.functional_causal import FunctionalCausalRecipe
from r0.modules.priors.lib.functional_causal_evaluation import functional_rng
from r0.modules.priors.lib.functional_target_calibration import FunctionalTargetCalibration
from r0.modules.priors.lib.linear import LinearRecipe
from r0.modules.priors.lib.population_generation_identity import preflight_calibrated_generation
from r0.modules.priors.lib.population_target_calibration import CalibratedPopulationGraph
from r0.modules.priors.lib.recursive_population_graph_output import (
    generate_recursive_population_graph,
)
from r0.modules.priors.lib.representatives import encode, record_fingerprint
from r0.modules.priors.lib.targets import adapt_targets
from r0.modules.priors.lib.training_task import TrainingTask


def generate_calibrated_population(
    recipe: CalibratedPopulationGraph, *, seed: int, task_id: int, device: str = "cpu"
):
    preflight_calibrated_generation(recipe, seed=seed, task_id=task_id, device=device)
    recipe.validate()
    original = generate_recursive_population_graph(
        recipe.original, seed=seed, task_id=task_id, device=device
    )
    scores, targets, probabilities, classes = apply_population_calibration(
        original,
        recipe.calibration,
        adapter=recipe.task
        if isinstance(recipe.original.request.base, FunctionalCausalRecipe)
        else None,
        rng=lambda key: functional_rng(seed, task_id, key),
    )
    config = asdict(recipe)
    record = json.loads(original.recipe_json)
    record["recipe"] = config
    record["execution"]["implementation"] = recipe.version
    source = record_fingerprint((config, seed, task_id))
    return replace(
        original,
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
                "original": json.loads(original.mechanism_json),
                "calibration": asdict(recipe.calibration),
                "calibration_identity": {
                    "seed": recipe.calibration_population.seed,
                    "task_id": recipe.calibration_population.task_id,
                },
                "query_fit": False,
            }
        ),
    )


def apply_population_calibration(
    original: TrainingTask,
    calibration: FunctionalTargetCalibration,
    *,
    adapter: LinearRecipe | None,
    rng: Callable[[str], np.random.Generator],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None, tuple[str, ...]]:
    """Apply retained constants to complete runtime rows, then the existing target adapter.

    Reference/intervention callers supply their matched or fresh stream mapping;
    this helper neither fits constants nor creates another calibration population.
    """
    scores = calibration.apply(original.scores)
    if adapter is not None:
        targets, probabilities, classes = adapt_targets(
            adapter, scores, noise_rng=rng("noise"), label_rng=rng("labels")
        )
    else:
        if original.classes:
            raise ValueError("direct category labels cannot receive numeric calibration")
        targets, probabilities, classes = scores[:, 0], None, original.classes
    return scores, targets, probabilities, classes
