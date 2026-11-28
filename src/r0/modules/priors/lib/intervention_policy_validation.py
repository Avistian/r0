"""Semantic replay checks shared by sampled-policy traces and persisted pairs."""

import json
from typing import Any, cast

import torch

from r0.modules.priors.lib.exact_frozen_record import exact_frozen_equal
from r0.modules.priors.lib.training_task import TrainingTask


def validate_policy_task(task: TrainingTask, replay: TrainingTask) -> None:
    for name in (
        "source_task_id",
        "episode_id",
        "task_type",
        "classes",
        "feature_schema",
        "target_origin",
    ):
        if not exact_frozen_equal(getattr(replay, name), getattr(task, name)):
            raise ValueError("policy task semantic metadata differs from replay")
    actual, expected = json.loads(task.recipe_json), json.loads(replay.recipe_json)
    execution = actual.get("execution")
    if type(execution) is not dict:
        raise ValueError("policy task execution must be a complete record")
    execution = cast(dict[str, Any], execution)
    if (
        set(execution) != set(expected["execution"])
        or execution.get("implementation") != expected["execution"]["implementation"]
        or type(execution.get("implementation")) is not str
        or execution.get("device") not in ("cpu", "cuda")
        or any(type(execution.get(k)) is not str for k in ("python", "numpy", "torch"))
        or type(execution.get("cpu_threads")) is not int
        or execution["cpu_threads"] < 1
        or any(
            execution.get(k) is not None and type(execution.get(k)) is not str
            for k in ("cuda", "gpu")
        )
        or set(actual) != set(expected)
        or any(
            json.dumps(actual[k], sort_keys=True, allow_nan=False)
            != json.dumps(expected[k], sort_keys=True, allow_nan=False)
            for k in expected
            if k != "execution"
        )
        or json.dumps(json.loads(task.mechanism_json), sort_keys=True, allow_nan=False)
        != json.dumps(json.loads(replay.mechanism_json), sort_keys=True, allow_nan=False)
    ):
        raise ValueError("policy task execution or archive differs from replay")
    for name in (
        "inputs",
        "targets",
        "scores",
        "context_indices",
        "query_indices",
        "probabilities",
    ):
        left, right = getattr(task, name), getattr(replay, name)
        if (
            (left is None) != (right is None)
            or left is not None
            and (left.dtype != right.dtype or not torch.equal(left.cpu(), right.cpu()))
        ):
            raise ValueError("policy task arrays differ from replay")
