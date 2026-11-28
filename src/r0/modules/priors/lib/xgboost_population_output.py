"""Fixed model execution keeps full fit lineage and never invokes XGBoost training."""

import json
from dataclasses import asdict, replace

from r0.modules.priors.lib.representatives import encode, record_fingerprint
from r0.modules.priors.lib.xgboost_population_graph import XGBoostPopulationGraph


def generate_xgboost_population_graph(
    recipe: XGBoostPopulationGraph, *, seed: int, task_id: int, device: str = "cpu"
):
    from r0.modules.priors.lib.generate import generate_task

    if device != "cpu" or any(type(v) is not int or not 0 <= v < 2**63 for v in (seed, task_id)):
        raise ValueError("population XGBoost requires exact identity and CPU")
    if recipe.population != "independent-ancestors" and (seed, task_id) != (
        recipe.fit_seed,
        recipe.fit_task_id,
    ):
        raise ValueError(
            "support/batch XGBoost archive is identity-locked; construct a new fit explicitly"
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
