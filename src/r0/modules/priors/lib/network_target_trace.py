"""Original target-adapter innovations shared by standalone networks and input views."""

import json
from collections.abc import Callable, Sequence
from dataclasses import asdict
from typing import Any, cast

import torch

from r0.modules.priors.lib.exact_frozen_record import exact_frozen_equal
from r0.modules.priors.lib.linear import LinearRecipe
from r0.modules.priors.lib.training_task import TrainingTask


def network_trace_record(task: TrainingTask) -> dict[str, Any]:
    raw = json.loads(task.recipe_json)
    if type(raw) is not dict:
        raise ValueError("network trace replay requires an object record")
    record = cast(dict[str, Any], raw)
    if any(
        type(record.get(key)) is not int or not 0 <= record[key] < 2**63
        for key in ("seed", "task_id")
    ):
        raise ValueError("network trace replay requires exact bounded integer identities")
    return record


def validate_network_trace(task: TrainingTask, replay: TrainingTask) -> None:
    """Validate semantic fields; execution environment remains historical provenance."""
    for field in (
        "source_task_id",
        "episode_id",
        "task_type",
        "classes",
        "feature_schema",
        "target_origin",
    ):
        if not exact_frozen_equal(getattr(task, field), getattr(replay, field)):
            raise ValueError("network trace metadata differs from original-row replay")
    if json.dumps(json.loads(task.mechanism_json), sort_keys=True, allow_nan=False) != json.dumps(
        json.loads(replay.mechanism_json), sort_keys=True, allow_nan=False
    ):
        raise ValueError("network trace mechanism metadata differs from replay")
    actual = network_trace_record(task).get("execution")
    expected = json.loads(replay.recipe_json)["execution"]
    if (
        type(actual) is not dict
        or cast(dict[str, Any], actual).get("implementation") != expected["implementation"]
    ):
        raise ValueError("network trace execution implementation differs from replay")
    for field in (
        "inputs",
        "targets",
        "scores",
        "context_indices",
        "query_indices",
        "probabilities",
    ):
        value, correct = getattr(task, field), getattr(replay, field)
        if value is None or correct is None:
            if value is not correct:
                raise ValueError("network trace probability presence differs from replay")
        elif (
            not isinstance(value, torch.Tensor)
            or value.dtype != correct.dtype
            or not torch.equal(value.cpu(), correct.cpu())
        ):
            raise ValueError("network trace arrays differ from original-row replay")


def network_target_steps(
    controls: LinearRecipe,
    task: TrainingTask,
    replay: TrainingTask,
    indices: Sequence[int],
    add: Callable[..., None],
) -> float:
    from r0.modules.priors.lib.functional_causal_evaluation import functional_rng
    from r0.modules.priors.lib.generation_graph import target_steps

    record = json.loads(task.recipe_json)

    def rng(stream: str):
        return functional_rng(record["seed"], record["task_id"], stream)

    if controls.task_type == "regression":
        add(
            "noise-draw",
            "Original standard normal draw",
            [],
            "independently keyed standard Gaussian observation draw",
            torch.tensor(
                rng("noise").normal(size=controls.rows)[list(indices)], dtype=torch.float64
            ),
            {"stream": "noise", "row_indices": list(indices)},
        )

    def adapter_add(
        key: str,
        label: str,
        parents: list[str],
        formula: str,
        values: torch.Tensor,
        parameters: dict[str, Any],
    ) -> None:
        if key == "noise":
            parents, formula = ["noise-draw"], "noise_ratio * original standard normal draw"
        add(key, label, parents, formula, values, parameters)

    targets = target_steps(
        asdict(controls), replay.scores[list(indices)], indices, rng, adapter_add
    )
    original = task.targets.cpu()[list(indices)]
    if not torch.equal(targets, original):
        raise ValueError("network target adapter differs from original-row replay")
    add("targets", "Task targets", ["target"], "shared target adapter output", targets[:, None], {})
    return float((targets - original).abs().max())
