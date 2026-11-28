"""Outer saved-task identity validation before any population fit replay."""

from typing import Any

from r0.modules.priors.lib.causal_limits import CausalLimits
from r0.modules.priors.lib.population_task_semantics import (
    population_task_record,
    validate_population_record_metadata,
    validate_population_task,
)
from r0.modules.priors.lib.representatives import record_fingerprint
from r0.modules.priors.lib.training_task import TrainingTask
from r0.modules.priors.lib.xgboost_population_graph_decoding import (
    preflight_xgboost_population_graph,
    xgboost_population_graph_from_dict,
)
from r0.modules.priors.lib.xgboost_population_output import generate_xgboost_population_graph


def preflight_xgboost_population_record(record: Any):
    record = validate_population_record_metadata(record)
    raw = record["recipe"]
    if (
        raw.get("version") != "population-xgboost-graph-v1"
        or record["execution"]["implementation"] != raw["version"]
        or record["execution"]["device"] != "cpu"
    ):
        raise ValueError("population XGBoost saved execution differs")
    preflight_xgboost_population_generation(raw, seed=record["seed"], task_id=record["task_id"])
    return record


def validate_xgboost_population_task(task: TrainingTask, *, limits: CausalLimits):
    record = preflight_xgboost_population_record(population_task_record(task))
    raw = record["recipe"]
    preflight_xgboost_population_graph(raw, limits=limits)
    controls = raw["base"]["task"]
    source = record_fingerprint((raw, record["seed"], record["task_id"]))
    episode = record_fingerprint((source, controls["split"], controls["context_rows"]))
    if (
        task.source_task_id != source
        or task.episode_id != episode
        or task.task_type != controls["task_type"]
    ):
        raise ValueError("population XGBoost task identity metadata differs")
    recipe = xgboost_population_graph_from_dict(raw, limits=limits)
    expected = generate_xgboost_population_graph(
        recipe, seed=record["seed"], task_id=record["task_id"]
    )
    validate_population_task(task, expected, record)
    return recipe


def preflight_xgboost_population_generation(
    raw: dict[str, Any], *, seed: Any, task_id: Any, count: Any = 1, device: Any = "cpu"
) -> None:
    if (
        any(type(v) is not int or not 0 <= v < 2**63 for v in (seed, task_id))
        or type(count) is not int
        or count < 1
        or device != "cpu"
    ):
        raise ValueError("population XGBoost generation requires exact identity and CPU")
    if raw.get("population") not in (
        "support-only",
        "batch-refit-reference",
        "independent-ancestors",
    ):
        raise ValueError("population XGBoost generation requires explicit lifecycle")
    if raw["population"] != "independent-ancestors" and (
        type(raw.get("fit_seed")) is not int
        or type(raw.get("fit_task_id")) is not int
        or (seed, task_id, count) != (raw["fit_seed"], raw["fit_task_id"], 1)
    ):
        raise ValueError("population XGBoost generation differs from retained fit episode")
