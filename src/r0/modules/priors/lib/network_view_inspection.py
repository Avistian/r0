"""Original observed coordinates, input masking and retained network architecture."""

import json
from collections.abc import Callable, Sequence
from typing import Any

import torch

from r0.modules.priors.lib.causal_limits import CausalLimits
from r0.modules.priors.lib.network_input_view import (
    generate_network_view,
    network_view_from_dict,
)
from r0.modules.priors.lib.network_inspection import network_steps
from r0.modules.priors.lib.training_task import TrainingTask


def network_view_steps(
    task: TrainingTask,
    indices: Sequence[int],
    add: Callable[..., None],
    *,
    limits: CausalLimits | None,
) -> dict[str, Any]:
    record = json.loads(task.recipe_json)
    view = network_view_from_dict(record["recipe"], limits=limits or CausalLimits())
    replay = generate_network_view(view, seed=record["seed"], task_id=record["task_id"])
    for field in ("inputs", "targets", "scores", "context_indices", "query_indices"):
        if not torch.equal(getattr(replay, field), getattr(task, field).cpu()):
            raise ValueError("saved network view differs from original-row replay")
    add(
        "view-observed",
        "Observed coordinates",
        [],
        "original joint root plus independently keyed extra columns",
        replay.inputs[list(indices)],
        {"root_width": view.base.function.input_width},
    )
    width = view.base.function.input_width
    mask = torch.zeros(width, dtype=torch.bool)
    mask[list(view.active_inputs)] = True
    argument = torch.where(mask, replay.inputs[:, :width], 0.0)
    add(
        "view-mask",
        "Network input mask",
        ["view-observed"],
        "keep selected original coordinates; fill inactive coordinates with zero",
        argument[list(indices)],
        {"active_inputs": view.active_inputs, "fill": 0.0},
    )
    parents = network_steps(
        view.base.function,
        argument,
        prefix="network",
        indices=indices,
        add=add,
        input_parents=["view-mask"],
    )
    add(
        "signal",
        "View scores",
        parents,
        "frozen network on masked inputs",
        replay.scores[list(indices)],
        {},
    )
    add(
        "targets",
        "Task targets",
        ["signal"],
        "shared target adapter",
        replay.targets[list(indices), None],
        {},
    )
    return {
        "version": "generation-graph-v1",
        "episode_id": task.episode_id,
        "rows": list(indices),
        "component": None,
        "component_count": 0,
        "signal_max_error": 0.0,
        "architecture": "frozen network with explicit observed-input mask",
    }
