"""Bounded slices through a saved function, keeping other inputs fixed."""

import json
import math
from typing import Any

import numpy as np
import torch

from r0.modules.priors import TrainingTask, evaluate_signal


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
    if type(feature) is not int or not 0 <= feature < task.inputs.shape[1]:
        raise ValueError("feature must identify an existing input column")
    if type(anchor_row) is not int or not 0 <= anchor_row < len(task.inputs):
        raise ValueError("anchor_row must identify an existing row")
    if type(points) is not int or not 2 <= points <= 512:
        raise ValueError("points must be 2..512")
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
    scores = evaluate_signal(task, inputs)
    mechanism = json.loads(task.mechanism_json)
    return {
        "version": "response-curve-v1",
        "kind": "analytic_response",
        "episode_id": task.episode_id,
        "feature": feature,
        "feature_role": "active" if feature in mechanism["active"] else "irrelevant",
        "anchor_row": anchor_row,
        "anchor_inputs": np.asarray(anchor.cpu().numpy(), dtype=np.float64).tolist(),
        "bounds": [low, high],
        "range_source": "observed_min_max" if bounds is None else "explicit",
        "points": points,
        "values": np.asarray(values.cpu().numpy(), dtype=np.float64).tolist(),
        "scores": np.asarray(scores.detach().cpu().numpy(), dtype=np.float64).tolist(),
    }
