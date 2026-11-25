"""Full-task statistics and explicitly bounded row samples, without mutation."""

import json
import math
from collections.abc import Sequence
from typing import Any

import numpy as np
from numpy.typing import NDArray

from r0.modules.priors import TrainingTask, record_fingerprint


def _summary(values: NDArray[np.float64]) -> dict[str, Any]:
    if not np.isfinite(values).all() or not len(values):
        raise ValueError("inspection requires nonempty finite values")
    magnitude = float(np.max(np.abs(values)))
    scaled = values / magnitude if magnitude else values
    low, high = float(scaled.min()), float(scaled.max())
    if low == high:
        counts, edges = [len(values)], [float(values[0]), float(values[0])]
    else:
        counts_array, edges_array = np.histogram(scaled, bins=20, range=(low, high))
        counts, edges = counts_array.tolist(), (edges_array * magnitude).tolist()
    mean, std = float(scaled.mean()) * magnitude, float(scaled.std()) * magnitude
    return {
        "count": len(values),
        "min": float(values.min()),
        "max": float(values.max()),
        "mean": mean if math.isfinite(mean) else None,
        "std": std if math.isfinite(std) else None,
        "quantiles": (np.quantile(scaled, [0, 0.25, 0.5, 0.75, 1]) * magnitude).tolist(),
        "histogram": {"counts": counts, "edges": edges},
    }


def inspect_rows(task: TrainingTask, indices: Sequence[int]) -> list[dict[str, Any]]:
    """Inspect original row identities, including diagnostic truth and split membership."""
    if not 1 <= len(indices) <= 2048 or any(
        type(i) is not int or not 0 <= i < len(task.inputs) for i in indices
    ):
        raise ValueError("select 1..2048 valid integer row indices")
    context = set(np.asarray(task.context_indices.detach().cpu().numpy(), dtype=np.int64).tolist())
    inputs = np.asarray(task.inputs.detach().cpu().numpy(), dtype=np.float64)
    targets = np.asarray(task.targets.detach().cpu().numpy(), dtype=np.float64)
    scores = np.asarray(task.scores.detach().cpu().numpy(), dtype=np.float64)
    probabilities = (
        None
        if task.probabilities is None
        else np.asarray(task.probabilities.detach().cpu().numpy(), dtype=np.float64)
    )
    return [
        {
            "index": i,
            "role": "context" if i in context else "query",
            "inputs": inputs[i].tolist(),
            "target": float(targets[i]) if task.task_type == "regression" else int(targets[i]),
            "scores": scores[i].tolist(),
            "probabilities": None if probabilities is None else probabilities[i].tolist(),
            "noise": float(targets[i] - scores[i, 0]) if task.task_type == "regression" else None,
        }
        for i in indices
    ]


def inspect_task(task: TrainingTask, *, sample_rows: int = 512) -> dict[str, Any]:
    """Summaries use every generated row; costly chart/table rows use a stable sample."""
    if type(sample_rows) is not int or not 1 <= sample_rows <= 2048:
        raise ValueError("sample_rows must be 1..2048")
    record = json.loads(task.recipe_json)
    mechanism = json.loads(task.mechanism_json)
    x = np.asarray(task.inputs.detach().cpu().numpy(), dtype=np.float64)
    y = np.asarray(task.targets.detach().cpu().numpy(), dtype=np.float64)
    scores = np.asarray(task.scores.detach().cpu().numpy(), dtype=np.float64)
    sample_seed = int(
        record_fingerprint(("explorer-rows-v1", record["seed"], record["task_id"], len(x))), 16
    )
    indices = sorted(
        np.random.default_rng(sample_seed)
        .choice(len(x), min(sample_rows, len(x)), replace=False)
        .tolist()
    )
    active = set(mechanism["active"])
    view: dict[str, Any] = {
        "version": "inspection-v1",
        "source_task_id": task.source_task_id,
        "episode_id": task.episode_id,
        "task_type": task.task_type,
        "classes": list(task.classes),
        "recipe": record,
        "family": mechanism["family"],
        "total_rows": len(x),
        "shown_rows": len(indices),
        "summary_scope": "all_rows",
        "sampling": "uniform without replacement, keyed by seed/task/row count",
        "split_counts": {"context": len(task.context_indices), "query": len(task.query_indices)},
        "features": [
            {
                "name": f"x{i}",
                "index": i,
                "role": "active" if i in active else "irrelevant",
                "summary": _summary(x[:, i]),
            }
            for i in range(x.shape[1])
        ],
        "target": _summary(y),
        "signals": [_summary(scores[:, c]) for c in range(scores.shape[1])],
        "rows": inspect_rows(task, indices),
        "classification": None,
        "noise": None,
    }
    if task.probabilities is None:
        view["noise"] = _summary(y - scores[:, 0])
    else:
        probabilities = np.asarray(task.probabilities.detach().cpu().numpy(), dtype=np.float64)
        with np.errstate(divide="ignore", invalid="ignore"):
            entropy = -np.where(probabilities > 0, probabilities * np.log(probabilities), 0).sum(
                axis=1
            )
        view["classification"] = {
            "observed_counts": np.bincount(
                y.astype(np.int64), minlength=len(task.classes)
            ).tolist(),
            "expected_counts": probabilities.sum(axis=0).tolist(),
            "context_counts": np.bincount(
                y[task.context_indices.cpu().numpy()].astype(np.int64), minlength=len(task.classes)
            ).tolist(),
            "query_counts": np.bincount(
                y[task.query_indices.cpu().numpy()].astype(np.int64), minlength=len(task.classes)
            ).tolist(),
            "entropy": _summary(entropy),
            "maximum_probability": _summary(probabilities.max(axis=1)),
        }
    return view
