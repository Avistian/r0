"""Emit fresh query rows with inherited independent calibration and complete source custody."""

import json
from dataclasses import asdict, replace

from r0.modules.priors.lib.calibrated_population_query import CalibratedPopulationQuery
from r0.modules.priors.lib.functional_causal import FunctionalCausalRecipe
from r0.modules.priors.lib.functional_causal_evaluation import functional_rng
from r0.modules.priors.lib.population_calibrated_output import apply_population_calibration
from r0.modules.priors.lib.population_generation_identity import preflight_calibrated_generation
from r0.modules.priors.lib.recursive_population_graph_output import (
    generate_recursive_population_graph,
)
from r0.modules.priors.lib.representatives import encode, record_fingerprint


def generate_calibrated_query(
    recipe: CalibratedPopulationQuery, *, seed: int, task_id: int, device: str = "cpu"
):
    preflight_calibrated_generation(recipe, seed=seed, task_id=task_id, device=device)
    recipe.validate()
    raw = generate_recursive_population_graph(
        recipe.original, seed=seed, task_id=task_id, device=device
    )
    scores, targets, probabilities, classes = apply_population_calibration(
        raw,
        recipe.calibration,
        adapter=recipe.task
        if isinstance(recipe.original.request.base, FunctionalCausalRecipe)
        else None,
        rng=lambda key: functional_rng(seed, task_id, key),
    )
    config = asdict(recipe)
    record = json.loads(raw.recipe_json)
    record["recipe"] = config
    record["execution"]["implementation"] = recipe.version
    source_id = record_fingerprint((config, seed, task_id))
    return replace(
        raw,
        source_task_id=source_id,
        episode_id=record_fingerprint((source_id, recipe.task.split, recipe.task.context_rows)),
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
                    "seed": recipe.calibration_population.seed,
                    "task_id": recipe.calibration_population.task_id,
                },
                "source_fingerprint": record_fingerprint(asdict(recipe.source)),
                "calibration_lifecycle": "retained-source-fit-new-query-only",
                "query_fit": False,
            }
        ),
    )
