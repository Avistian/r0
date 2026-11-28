"""Detailed frozen runtime trace plus separately scoped actual fit populations."""

from collections.abc import Sequence
from typing import Any

from r0.modules.priors.lib.causal_limits import CausalLimits
from r0.modules.priors.lib.population_task_semantics import population_task_record
from r0.modules.priors.lib.training_task import TrainingTask
from r0.modules.priors.lib.xgboost_population_task import validate_xgboost_population_task


def explain_xgboost_population(
    task: TrainingTask,
    indices: Sequence[int],
    *,
    component: int | None = None,
    column_start: int = 0,
    limits: CausalLimits | None = None,
) -> dict[str, Any]:
    from r0.modules.priors.lib.generate import generate_task
    from r0.modules.priors.lib.generation_graph import explain_generation

    if component is not None or type(column_start) is not int or not 0 <= column_start <= 4095:
        raise ValueError("population XGBoost inspection needs whole graph and bounded columns")
    if not 1 <= len(indices) <= 16 or any(
        type(i) is not int or not 0 <= i < len(task.inputs) for i in indices
    ):
        raise ValueError("population XGBoost invalid original row indices")
    recipe = validate_xgboost_population_task(task, limits=limits or CausalLimits())
    record = population_task_record(task)
    runtime = generate_task(recipe.runtime, seed=record["seed"], task_id=record["task_id"])
    graph = explain_generation(runtime, indices, column_start=column_start, limits=limits)
    for fit in recipe.fits:
        for label, values in (
            ("inputs", fit.realization.inputs),
            ("fake-targets", fit.realization.targets),
        ):
            width = len(values[0])
            start = min(column_start, width - 1)
            end = min(start + 32, width)
            graph["nodes"].append(
                dict(
                    id=f"fit-{fit.node}-{label}",
                    label=f"Node{fit.node} retained fitting {label}",
                    parents=[],
                    formula="Actual synthetic parent fit population; frozen during query replay",
                    shape=[len(values), width],
                    column_indices=list(range(start, end)),
                    values=[list(row[start:end]) for row in values[:32]],
                    parameters={
                        "row_labels": list(fit.row_ids[:32]),
                        "row_heading": "Original fitting population row",
                        "axes": "Retained fit population; independent of displayed query rows",
                        "scope": "fit-population",
                        "population": recipe.population,
                        "fit_seed": recipe.fit_seed,
                        "fit_task_id": recipe.fit_task_id,
                        "total_fit_rows": len(values),
                        "query_refit": False,
                    },
                )
            )
    return {
        **graph,
        "kind": "population_xgboost",
        "episode_id": task.episode_id,
        "population": recipe.population,
        "query_refit": False,
    }
