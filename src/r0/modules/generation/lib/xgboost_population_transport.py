"""Population XGBoost run headers and producer custody before source replay."""

import json
from dataclasses import asdict
from typing import TYPE_CHECKING, Any, cast

from r0.modules.priors import CausalLimits, LinearRecipe, XGBoostPopulationGraph
from r0.modules.priors.lib.xgboost_population_task import (
    preflight_xgboost_population_generation,
    preflight_xgboost_population_record,
)

if TYPE_CHECKING:
    from r0.modules.generation.lib.requests import RunRequest


def validate_xgboost_population_run(request: "RunRequest") -> None:
    if isinstance(request.recipe, XGBoostPopulationGraph):
        preflight_xgboost_population_generation(
            asdict(request.recipe),
            seed=request.seed,
            task_id=request.start,
            count=request.count,
            device=request.device,
        )
        if request.sampling is not None or request.construction is not None:
            raise ValueError("population XGBoost replay cannot sample or construct")


def preflight_xgboost_population_run(
    raw: Any, fields: dict[str, Any], limits: CausalLimits | None, construction: Any
) -> None:
    from r0.modules.generation.lib.requests import RunRequest

    if (
        type(raw) is dict
        and cast(dict[str, Any], raw).get("version") == "population-xgboost-graph-v1"
    ):
        preflight_xgboost_population_generation(
            cast(dict[str, Any], raw),
            seed=fields.get("seed"),
            task_id=fields.get("start", 0),
            count=fields.get("count"),
            device=fields.get("device", "cpu"),
        )
        if construction is not None or fields.get("sampling") is not None:
            raise ValueError("population XGBoost replay cannot sample or construct")
        RunRequest(LinearRecipe(), causal_limits=limits, **fields)


def xgboost_population_record_matches(
    record: dict[str, Any], request: "RunRequest", selected: Any, task_id: int
) -> bool | None:
    if not isinstance(selected, XGBoostPopulationGraph):
        return None
    preflight_xgboost_population_record(record)
    return (
        json.dumps(record["recipe"], sort_keys=True, allow_nan=False)
        == json.dumps(asdict(selected), sort_keys=True, allow_nan=False)
        and record["seed"] == request.seed
        and record["task_id"] == task_id
    )
