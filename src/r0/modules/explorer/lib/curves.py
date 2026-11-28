"""Bounded slices through a saved function, keeping other inputs fixed."""

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import torch

from r0.modules.priors import CausalLimits, TrainingTask, condition_gp, evaluate_signal

_DEFAULT_LIMITS = CausalLimits()


def inspect_curve(
    task: TrainingTask,
    *,
    feature: int,
    anchor_row: int = 0,
    points: int = 129,
    bounds: tuple[float, float] | None = None,
    limits: CausalLimits = _DEFAULT_LIMITS,
) -> dict[str, Any]:
    """Evaluate an analytic response slice; never interpolate a direct GP draw.

    All columns except ``feature`` stay at the selected original row's values.
    The default range is that column's full observed minimum/maximum. Explicit
    bounds may extrapolate beyond the data. Scores exclude observation noise and
    classification separation/offsets; no new labels or random values are sampled.
    """
    inputs, metadata = _slice_inputs(task, feature, anchor_row, points, bounds, maximum=512)
    scores = evaluate_signal(task, inputs, limits=limits)
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


def _slice_metadata(
    task: TrainingTask,
    feature: int,
    anchor_row: int,
    points: int,
    bounds: tuple[float, float] | None,
    *,
    maximum: int,
) -> dict[str, Any]:
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
    mechanism = json.loads(task.mechanism_json)
    return {
        "episode_id": task.episode_id,
        "feature": feature,
        "feature_role": "active" if feature in mechanism["active"] else "irrelevant",
        "anchor_row": anchor_row,
        "anchor_inputs": np.asarray(anchor.cpu().numpy(), dtype=np.float64).tolist(),
        "bounds": [low, high],
        "range_source": "observed_min_max" if bounds is None else "explicit",
        "points": points,
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
    metadata = _slice_metadata(task, feature, anchor_row, points, bounds, maximum=maximum)
    low, high = metadata["bounds"]
    inputs = task.inputs[anchor_row].detach().repeat(points, 1)
    fraction = torch.linspace(0, 1, points, dtype=inputs.dtype, device=inputs.device)
    values = (1 - fraction) * low + fraction * high
    inputs[:, feature] = values
    metadata["values"] = np.asarray(values.cpu().numpy(), dtype=np.float64).tolist()
    return inputs, metadata


def inspect_network_curve_guarded(
    task: TrainingTask, data: dict[str, Any], output_dir: Path, *, limits: CausalLimits
) -> dict[str, Any]:
    from dataclasses import asdict

    from r0.modules.generation import run_guarded_document

    metadata = _slice_metadata(
        task,
        data["feature"],
        data.get("anchor_row", 0),
        data.get("points", 129),
        data.get("bounds"),
        maximum=512,
    )
    record = json.loads(task.recipe_json)
    noisy = record["recipe"].get("version") == "standalone-network-innovations-v1"
    functional = record["recipe"].get("version") in (
        "functional-causal-v1",
        "functional-causal-v2",
        "xgboost-fitted-functional-causal-v1",
        "sampled-hyperprior-graph-v1",
        "sampled-joint-hyperprior-graph-v1",
    )
    result = run_guarded_document(
        {
            "validation": "network-curve",
            "reference": data.get("reference", "original-row" if functional else "zero"),
            "seed": record["seed"],
            "task_id": record["task_id"],
            "anchor_row": metadata["anchor_row"],
            "recipe": json.loads(task.recipe_json)["recipe"],
            "anchor": metadata["anchor_inputs"],
            "feature": metadata["feature"],
            "points": metadata["points"],
            "bounds": metadata["bounds"],
            "causal_limits": asdict(limits),
            "max_seconds": data.get("max_seconds", 60),
            "max_bytes": data.get("max_bytes", 256 * 1024**2),
        },
        output_dir / "curve-validation",
    )
    if result.timed_out:
        raise ValueError("Network response curve timed out; saved dataset unchanged")
    if result.returncode or result.summary is None:
        detail = result.stderr.strip().splitlines()
        raise ValueError("Network response curve failed: " + (detail[-1] if detail else "failed"))
    return {
        **metadata,
        **result.summary,
        "version": "response-curve-v1",
        "kind": "functional_causal_reference"
        if functional
        else "network_innovation_reference"
        if noisy
        else "analytic_response",
        "score_semantics": (
            "raw target readouts under simultaneous state-and-readout clamps of observed channels; "
            "hidden roots fixed to the original row, structural draws fixed or zero; "
            "not a conditional mean, calibrated task score or enclosing task probability"
        )
        if functional
        else (
            "raw latent zero-innovation or original-row fixed-innovation reference; "
            "not conditional mean or class probability"
        )
        if noisy
        else "frozen network on observed batch coordinates; no root redraw or query normalization"
        if record["recipe"].get("version") == "prototype-batch-network-realized-v1"
        else (
            "frozen latent network channels, including retained matching; not class probabilities"
        ),
    }


def inspect_response_curve(
    task: TrainingTask, data: dict[str, Any], output_dir: Path
) -> dict[str, Any]:
    """Route saved construction references through guarded worker evaluation."""
    from r0.modules.explorer.lib.construction import caller_limits

    if json.loads(task.recipe_json)["recipe"].get("version") in (
        "functional-causal-v1",
        "functional-causal-v2",
        "xgboost-fitted-functional-causal-v1",
        "sampled-hyperprior-graph-v1",
        "sampled-joint-hyperprior-graph-v1",
        "prototype-batch-network-realized-v1",
        "standalone-network-v1",
        "standalone-network-innovations-v1",
        "network-input-view-v1",
        "network-input-view-v2",
    ):
        return inspect_network_curve_guarded(task, data, output_dir, limits=caller_limits(data))
    return inspect_curve(
        task,
        feature=data["feature"],
        anchor_row=data.get("anchor_row", 0),
        points=data.get("points", 129),
        bounds=data.get("bounds"),
    )
