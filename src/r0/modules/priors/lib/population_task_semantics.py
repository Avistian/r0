"""Strict population task semantics, separate from historical execution descriptions."""

import json
from typing import Any, cast

import torch

from r0.modules.priors.lib.exact_frozen_record import exact_frozen_equal
from r0.modules.priors.lib.recursive_population_records import population_identity
from r0.modules.priors.lib.representatives import encode
from r0.modules.priors.lib.training_task import TrainingTask

_RECORD_FIELDS = {"recipe", "seed", "task_id", "execution"}
_EXECUTION_FIELDS = {
    "implementation",
    "device",
    "python",
    "numpy",
    "torch",
    "cpu_threads",
    "cuda",
    "gpu",
}


def is_population_task_record(record: Any, card: Any) -> bool:
    if type(card) is dict and cast(dict[str, Any], card).get("family") == "recursive_population":
        return True
    if type(record) is not dict:
        return False
    recipe = cast(dict[str, Any], record).get("recipe")
    return type(recipe) is dict and cast(dict[str, Any], recipe).get("version") in (
        "recursive-population-v1",
        "recursive-population-intervention-v1",
    )


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("population record contains a duplicate JSON field")
        result[key] = value
    return result


def _json_record(text: str) -> Any:
    return json.loads(text, object_pairs_hook=_unique_object)


def population_task_record(task: TrainingTask) -> dict[str, Any]:
    """Validate the complete public record before accessing any required field."""
    if type(task) is not TrainingTask or type(task.recipe_json) is not str:
        raise ValueError("population task requires its typed semantic record")
    raw = _json_record(task.recipe_json)
    if type(raw) is not dict:
        raise ValueError("population record must be a complete object")
    record = cast(dict[str, Any], raw)
    if set(record) != _RECORD_FIELDS or type(record["recipe"]) is not dict:
        raise ValueError("population record requires exactly recipe, identities and execution")
    population_identity(record["seed"], record["task_id"])
    value = record["execution"]
    if type(value) is not dict:
        raise ValueError("population execution must be a complete object")
    execution = cast(dict[str, Any], value)
    if (
        set(execution) != _EXECUTION_FIELDS
        or any(
            type(execution.get(key)) is not str or not execution[key]
            for key in ("implementation", "device", "python", "numpy", "torch")
        )
        or execution["device"] not in ("cpu", "cuda")
        or type(execution.get("cpu_threads")) is not int
        or not 1 <= execution["cpu_threads"] < 2**63
        or any(
            execution.get(key) is not None
            and (type(execution[key]) is not str or not execution[key])
            for key in ("cuda", "gpu")
        )
    ):
        raise ValueError("population execution schema or primitive metadata is invalid")
    return record


def _same_tensor(actual: object, expected: torch.Tensor) -> bool:
    if type(actual) is not torch.Tensor:
        return False
    # Strides/storage offsets may legitimately differ after archive materialization.
    # Tensor layout, dtype, shape, device and gradient status remain semantic.
    return (
        actual.dtype == expected.dtype
        and actual.layout == expected.layout
        and actual.shape == expected.shape
        and actual.device == expected.device
        and actual.requires_grad == expected.requires_grad
        and torch.equal(actual, expected)
    )


def validate_population_task(
    task: TrainingTask, replay: TrainingTask, record: dict[str, Any]
) -> None:
    """Reject numeric/type aliases without rewriting historical environment provenance."""
    expected = population_task_record(replay)
    if (
        record["execution"]["implementation"] != expected["execution"]["implementation"]
        or encode({key: value for key, value in record.items() if key != "execution"})
        != encode({key: value for key, value in expected.items() if key != "execution"})
        or any(
            not exact_frozen_equal(getattr(task, key), getattr(replay, key))
            for key in (
                "source_task_id",
                "episode_id",
                "task_type",
                "classes",
                "feature_schema",
                "target_origin",
            )
        )
        or type(task.mechanism_json) is not str
        or encode(_json_record(task.mechanism_json)) != encode(_json_record(replay.mechanism_json))
        or task.probabilities is not None
        or any(
            not _same_tensor(getattr(task, key), getattr(replay, key))
            for key in ("inputs", "targets", "scores", "context_indices", "query_indices")
        )
    ):
        raise ValueError("population task differs from exact semantic replay")
