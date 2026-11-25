"""Bounded slices through a saved function, keeping other inputs fixed."""

import json
import math
from typing import Any

import numpy as np
import torch

from r0.modules.priors import TrainingTask, condition_gp, evaluate_signal


def inspect_curve(
    task: TrainingTask,
    *,
    feature: int,
    anchor_row: int = 0,
    points: int = 129,
    bounds: tuple[float, float] | None = None,
) -> dict[str, Any]:
    """Evaluate an analytic response slice; never interpolate a direct GP draw.

    All columns except ``feature`` stay at the selected original row's values.
    The default range is that column's full observed minimum/maximum. Explicit
    bounds may extrapolate beyond the data. Scores exclude observation noise and
    classification separation/offsets; no new labels or random values are sampled.
    """
    inputs, metadata = _slice_inputs(task, feature, anchor_row, points, bounds, maximum=512)
    scores = evaluate_signal(task, inputs)
    return {
        **metadata,
        "version": "response-curve-v1",
        "kind": "analytic_response",
        "scores": np.asarray(scores.detach().cpu().numpy(), dtype=np.float64).tolist(),
    }


def inspect_gp_curve(
    task: TrainingTask,
    *,
    feature: int,
    anchor_row: int = 0,
    points: int = 129,
    bounds: tuple[float, float] | None = None,
    draws: int = 0,
    seed: int | None = None,
    allow_rank_truncation: bool = False,
) -> dict[str, Any]:
    """Conditional GP slice with separately identified original dataset values."""
    inputs, metadata = _slice_inputs(task, feature, anchor_row, points, bounds, maximum=256)
    result = condition_gp(
        task,
        inputs.cpu(),
        draws=draws,
        seed=seed,
        allow_rank_truncation=allow_rank_truncation,
    )
    observed = task.inputs.detach().cpu().numpy()
    card = json.loads(task.mechanism_json)
    fixed = [axis for axis in card["active"] if axis != feature]
    matches = np.all(observed[:, fixed] == observed[anchor_row, fixed], axis=1)
    low, high = metadata["bounds"]
    matches &= (observed[:, feature] >= low) & (observed[:, feature] <= high)
    indices = np.flatnonzero(matches)
    shown = indices[np.linspace(0, len(indices) - 1, min(len(indices), 512), dtype=np.int64)]
    return {
        **result,
        **metadata,
        "kind": "gp_conditional_curve",
        "original": {
            "meaning": "original dataset values with matching fixed active coordinates",
            "matched_count": len(indices),
            "shown_count": len(shown),
            "total_task_rows": len(task.inputs),
            "row_indices": shown.tolist(),
            "values": observed[shown, feature].tolist(),
            "scores": task.scores.detach().cpu().numpy()[shown].tolist(),
        },
    }


def _slice_inputs(
    task: TrainingTask,
    feature: int,
    anchor_row: int,
    points: int,
    bounds: tuple[float, float] | None,
    *,
    maximum: int,
) -> tuple[torch.Tensor, dict[str, Any]]:
    if type(feature) is not int or not 0 <= feature < task.inputs.shape[1]:
        raise ValueError("feature must identify an existing input column")
    if type(anchor_row) is not int or not 0 <= anchor_row < len(task.inputs):
        raise ValueError("anchor_row must identify an existing row")
    if type(points) is not int or not 2 <= points <= maximum:
        raise ValueError(f"points must be 2..{maximum}")
    if bounds is None:
        column = task.inputs[:, feature]
        low, high = float(column.min()), float(column.max())
    else:
        if len(bounds) != 2 or any(
            type(value) not in (int, float) or not math.isfinite(value) for value in bounds
        ):
            raise ValueError("bounds must contain two finite numbers")
        low, high = bounds
        if low > high:
            raise ValueError("bounds must be ordered from low to high")
    anchor = task.inputs[anchor_row].detach()
    inputs = anchor.repeat(points, 1)
    fraction = torch.linspace(0, 1, points, dtype=inputs.dtype, device=inputs.device)
    # Avoid high-low overflow for finite endpoints with opposite signs.
    values = (1 - fraction) * low + fraction * high
    inputs[:, feature] = values
    mechanism = json.loads(task.mechanism_json)
    return inputs, {
        "episode_id": task.episode_id,
        "feature": feature,
        "feature_role": "active" if feature in mechanism["active"] else "irrelevant",
        "anchor_row": anchor_row,
        "anchor_inputs": np.asarray(anchor.cpu().numpy(), dtype=np.float64).tolist(),
        "bounds": [low, high],
        "range_source": "observed_min_max" if bounds is None else "explicit",
        "points": points,
        "values": np.asarray(values.cpu().numpy(), dtype=np.float64).tolist(),
    }
