"""Original observed coordinates, input masking and retained network architecture."""

import json
from collections.abc import Callable, Sequence
from typing import Any

import torch

from r0.modules.priors.lib.causal_limits import CausalLimits
from r0.modules.priors.lib.frozen_function_evaluation import evaluate_admitted_function
from r0.modules.priors.lib.matched_network_view import (
    MatchedNetworkInputView,
    matched_view_from_dict,
)
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
    decoder = (
        matched_view_from_dict
        if record["recipe"].get("version") == "network-input-view-v2"
        else network_view_from_dict
    )
    view = decoder(record["recipe"], limits=limits or CausalLimits())
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
    if isinstance(view, MatchedNetworkInputView):
        assert view.matching is not None
        matching = view.matching
        raw = evaluate_admitted_function(view.base.function, argument)
        add(
            "view-raw-scores",
            "Raw masked network scores",
            parents,
            "frozen network before independent matching",
            raw[list(indices)],
            {},
        )
        add(
            "view-matching",
            "Estimated baseline moment matching",
            ["view-raw-scores"],
            "(raw - after_mean) / after_std * before_std + before_mean",
            matching.apply(raw)[list(indices)],
            {
                "population_digest": matching.digest,
                "calibration_rows": matching.rows,
                "before_mean": matching.before_mean,
                "before_std": matching.before_std,
                "after_mean": matching.after_mean,
                "after_std": matching.after_std,
                "scope": "independent estimate; not unit variance or episode normalization",
            },
        )
        parents = ["view-matching"]
    add(
        "signal",
        "View scores",
        parents,
        (
            "independently matched network on masked inputs"
            if isinstance(view, MatchedNetworkInputView)
            else "frozen network on masked inputs"
        ),
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
