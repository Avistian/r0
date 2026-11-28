"""Fixed model execution keeps full fit lineage and never invokes XGBoost training."""

import json
from dataclasses import asdict, replace

from r0.modules.priors.lib.representatives import encode, record_fingerprint
from r0.modules.priors.lib.xgboost_population_graph import XGBoostPopulationGraph


def generate_xgboost_population_graph(
    recipe: XGBoostPopulationGraph, *, seed: int, task_id: int, device: str = "cpu"
):
    from r0.modules.priors.lib.generate import generate_task
    from r0.modules.priors.lib.xgboost_population_task import (
        preflight_xgboost_population_generation,
    )

    preflight_xgboost_population_generation(
        {
            "population": recipe.population,
            "fit_seed": recipe.fit_seed,
            "fit_task_id": recipe.fit_task_id,
        },
        seed=seed,
        task_id=task_id,
        device=device,
    )
    recipe.validate()
    task = generate_task(recipe.runtime, seed=seed, task_id=task_id, device=device)
    record = json.loads(task.recipe_json)
    record["recipe"] = asdict(recipe)
    record["execution"]["implementation"] = recipe.version
    identity = record_fingerprint((asdict(recipe), seed, task_id))
    card = {
        "family": "population_xgboost",
        "version": recipe.version,
        "population": recipe.population,
        "fit_seed": recipe.fit_seed,
        "fit_task_id": recipe.fit_task_id,
        "query_refit": False,
        "runtime": json.loads(task.mechanism_json),
        "fit_rows": [list(f.row_ids) for f in recipe.fits],
    }
    return replace(
        task,
        recipe_json=encode(record),
        mechanism_json=encode(card),
        source_task_id=identity,
        episode_id=record_fingerprint((identity, recipe.task.split, recipe.task.context_rows)),
    )
