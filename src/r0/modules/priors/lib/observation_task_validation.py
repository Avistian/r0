"""Bind saved observation tasks to complete semantic replay before any trace slicing."""

import json
from typing import Any, cast

import torch

from r0.modules.priors.lib.causal_limits import CausalLimits
from r0.modules.priors.lib.exact_frozen_record import exact_frozen_equal
from r0.modules.priors.lib.training_task import TrainingTask


def validate_observation_task(
    task: TrainingTask, record: dict[str, Any], *, limits: CausalLimits
) -> None:
    from r0.modules.priors.lib.generate import generate_task, recipe_from_dict

    if any(
        type(record.get(key)) is not int or not 0 <= record[key] < 2**63
        for key in ("seed", "task_id")
    ):
        raise ValueError("observation trace requires exact bounded integer identities")
    execution = record.get("execution")
    keys = {"implementation", "device", "python", "numpy", "torch", "cpu_threads", "cuda", "gpu"}
    if type(execution) is not dict:
        raise ValueError("observation trace execution schema differs")
    execution = cast(dict[str, Any], execution)
    if (
        set(execution) != keys
        or execution.get("implementation") != record["recipe"]["version"]
        or execution.get("device") not in ("cpu", "cuda")
        or any(type(execution.get(key)) is not str for key in ("python", "numpy", "torch"))
        or type(execution.get("cpu_threads")) is not int
        or execution["cpu_threads"] < 1
        or any(
            execution.get(key) is not None and type(execution.get(key)) is not str
            for key in ("cuda", "gpu")
        )
    ):
        raise ValueError("observation trace execution semantics differ")
    # Strict source admission precedes replay. Original tensors are not copied or
    # scanned before their shapes/dtypes/layouts have matched the admitted replay.
    recipe = recipe_from_dict(record["recipe"], limits=limits)
    replay = generate_task(recipe, seed=record["seed"], task_id=record["task_id"])
    for field in (
        "source_task_id",
        "episode_id",
        "task_type",
        "classes",
        "feature_schema",
        "target_origin",
    ):
        if not exact_frozen_equal(getattr(task, field), getattr(replay, field)):
            raise ValueError("observation trace semantic metadata differs from replay")
    expected = json.loads(replay.recipe_json)
    if (
        set(record) != set(expected)
        or any(
            json.dumps(record[key], sort_keys=True, allow_nan=False)
            != json.dumps(expected[key], sort_keys=True, allow_nan=False)
            for key in expected
            if key != "execution"
        )
        or json.dumps(json.loads(task.mechanism_json), sort_keys=True, allow_nan=False)
        != json.dumps(json.loads(replay.mechanism_json), sort_keys=True, allow_nan=False)
    ):
        raise ValueError("observation metadata or recipe provenance differs from replay")
    for field in (
        "inputs",
        "targets",
        "scores",
        "context_indices",
        "query_indices",
        "probabilities",
    ):
        actual, correct = getattr(task, field), getattr(replay, field)
        if actual is None or correct is None:
            if actual is not correct:
                raise ValueError("observation trace probability presence differs from replay")
        elif (
            not isinstance(actual, torch.Tensor)
            or actual.shape != correct.shape
            or actual.dtype != correct.dtype
            or actual.layout != correct.layout
            or not torch.equal(actual.cpu(), correct.cpu())
        ):
            raise ValueError("observation trace arrays differ from original-row replay")
