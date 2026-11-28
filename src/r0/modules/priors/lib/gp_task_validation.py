"""Exact semantic validation shared by retained GP task environments."""

import json
from typing import Any, cast

import torch

from r0.modules.priors.lib.gp_column_decoding import gp_payload_equal
from r0.modules.priors.lib.training_task import TrainingTask


def validate_gp_execution(execution: object, version: str) -> None:
    """Reject malformed execution headers before numerical replay."""
    if type(execution) is not dict:
        raise ValueError("GP execution metadata schema differs")
    execution = cast(dict[str, Any], execution)
    if (
        set(execution)
        != {"implementation", "device", "python", "numpy", "torch", "cpu_threads", "cuda", "gpu"}
        or execution.get("implementation") != version
        or execution.get("device") not in ("cpu", "cuda")
        or any(type(execution[k]) is not str for k in ("python", "numpy", "torch"))
        or type(execution["cpu_threads"]) is not int
        or execution["cpu_threads"] < 1
        or any(execution[k] is not None and type(execution[k]) is not str for k in ("cuda", "gpu"))
    ):
        raise ValueError("GP execution metadata schema differs")


def validate_gp_task(task: TrainingTask, expected: TrainingTask, version: str) -> None:
    record = json.loads(task.recipe_json)
    expected_record = json.loads(expected.recipe_json)
    if set(record) != set(expected_record) or any(
        not gp_payload_equal(v, expected_record[k]) for k, v in record.items() if k != "execution"
    ):
        raise ValueError("GP task provenance differs from replay")
    validate_gp_execution(record["execution"], version)
    if not gp_payload_equal(json.loads(task.mechanism_json), json.loads(expected.mechanism_json)):
        raise ValueError("GP mechanism metadata differs from replay")
    for field in (
        "source_task_id",
        "episode_id",
        "task_type",
        "classes",
        "feature_schema",
        "target_origin",
    ):
        if not gp_payload_equal(getattr(task, field), getattr(expected, field)):
            raise ValueError("GP task semantic metadata differs")
    for field in (
        "inputs",
        "targets",
        "scores",
        "probabilities",
        "context_indices",
        "query_indices",
    ):
        actual, wanted = getattr(task, field), getattr(expected, field)
        if actual is None or wanted is None:
            if actual is not wanted:
                raise ValueError("GP task probabilities differ")
        elif (
            actual.dtype != wanted.dtype
            or actual.layout != wanted.layout
            or not torch.equal(actual.cpu(), wanted)
        ):
            raise ValueError("GP task tensor differs from original population replay")
