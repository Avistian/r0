"""Full-task statistics and explicitly bounded row samples, without mutation."""

import json
import math
from collections.abc import Sequence
from dataclasses import asdict
from typing import Any

import numpy as np
from numpy.typing import NDArray

from r0.modules.explorer.lib.construction_details import construction_details
from r0.modules.priors import TrainingTask, record_fingerprint


def _sample_rows(total: int, limit: int, rng: np.random.Generator) -> list[int]:
    return sorted(rng.choice(total, min(limit, total), replace=False).tolist())


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
            **(
                {
                    "display_inputs": [
                        f.decode(float(v))
                        for f, v in zip(task.feature_schema, inputs[i], strict=True)
                    ],
                    "target_label": task.classes[int(targets[i])]
                    if task.classes
                    else float(targets[i]),
                }
                if task.feature_schema
                else {}
            ),
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
    indices = _sample_rows(len(x), sample_rows, np.random.default_rng(sample_seed))
    active = set(mechanism.get("active", ()))
    view: dict[str, Any] = {
        "version": "inspection-v1",
        "source_task_id": task.source_task_id,
        "episode_id": task.episode_id,
        "task_type": task.task_type,
        "classes": list(task.classes),
        "recipe": record,
        "family": mechanism["family"],
        "construction_details": construction_details(record["recipe"]),
        "total_rows": len(x),
        "shown_rows": len(indices),
        "summary_scope": "all_rows",
        "sampling": "uniform without replacement, keyed by seed/task/row count",
        "split_counts": {"context": len(task.context_indices), "query": len(task.query_indices)},
        "features": [
            {
                "name": f"x{i}",
                "index": i,
                "role": "joint GP column"
                if mechanism["family"] == "gp_columns"
                else "active"
                if i in active
                else "irrelevant",
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
    if task.feature_schema:
        view["feature_schema"] = [asdict(f) for f in task.feature_schema]
        view["target_origin"] = task.target_origin
        native = record["recipe"]
        view["native_construction"] = native["version"]
        if native["version"] == "causal-intervention-v1":
            native = native["base"]
        if native["version"] in ("converted-causal-v1", "sampled-converted-graph-v1"):
            from r0.modules.explorer.lib.converted import converted_target_view

            view.update(converted_target_view(record))
        else:
            if native["version"] in ("sampled-native-graph-v1", "sampled-native-vector-graph-v1"):
                native = native["recipe"]
            target_node = native["nodes"][native["target"][0]]
            domain = (
                target_node["domains"][native["target"][1]]
                if native["version"] == "native-vector-causal-v1"
                else target_node["domain"]
            )
            original_domain = domain
            changes = record["recipe"].get("interventions", ())
            target_node_index, target_variable = native["target"]
            relevant = [
                c
                for c in changes
                if c["node"] == target_node_index
                and ("variable" not in c or c["variable"] == target_variable)
            ]
            for change in relevant:
                if change.get("domain") is not None:
                    domain = change["domain"]
            view["target_emission"] = domain["emission"]
            if domain != original_domain:
                view["original_target_emission"] = original_domain["emission"]
                view["original_target_frequency_weights"] = original_domain.get("frequencies")
                view["original_target_domain"] = original_domain
                view["effective_target_domain"] = domain
            if any(c.get("category") is not None for c in relevant):
                view["original_target_emission"] = original_domain["emission"]
                view["target_emission"] = "forced-category indicator"
            view["target_frequency_weights"] = domain.get("frequencies")
        for feature, schema in zip(view["features"], task.feature_schema, strict=True):
            feature["kind"] = schema.kind
            feature["role"] = "native observed variable"
            if schema.kind != "continuous":
                feature["summary"] = {
                    "categories": list(schema.vocabulary),
                    "counts": np.bincount(
                        x[:, feature["index"]].astype(np.int64), minlength=len(schema.vocabulary)
                    ).tolist(),
                    "count": len(x),
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
