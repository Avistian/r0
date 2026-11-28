"""Saved GP-root references dispatch all structural execution to the guard."""

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

from r0.modules.explorer.lib.construction import caller_limits
from r0.modules.generation import run_guarded_document
from r0.modules.priors import TrainingFeature, TrainingTask


def inspect_gp_root_curve(
    task: TrainingTask, data: dict[str, Any], output_dir: Path
) -> dict[str, Any]:
    feature, row, points = data["feature"], data.get("anchor_row", 0), data.get("points", 33)
    if (
        type(feature) is not int
        or not 0 <= feature < task.inputs.shape[1]
        or type(row) is not int
        or not 0 <= row < len(task.inputs)
        or type(points) is not int
        or not 2 <= points <= 129
    ):
        raise ValueError("GP reference requires a valid feature, row and 2..129 points")
    column = task.inputs[:, feature]
    low, high = float(column.min()), float(column.max())
    schema = task.feature_schema[feature] if task.feature_schema else TrainingFeature()
    values = (
        list(range(len(schema.vocabulary)))
        if schema.vocabulary
        else [(1 - i / (points - 1)) * low + i / (points - 1) * high for i in range(points)]
    )
    record = json.loads(task.recipe_json)
    result = run_guarded_document(
        {
            "validation": "gp-root-curve",
            "recipe": record["recipe"],
            "seed": record["seed"],
            "task_id": record["task_id"],
            "feature": feature,
            "anchor_row": row,
            "anchor": [float(v) for v in task.inputs[row]],
            "values": values,
            "inputs": data.get("inputs"),
            "reference": data.get("reference", "original-row"),
            "capture_location": data.get("capture_location", "capture"),
            "causal_limits": asdict(caller_limits(data)),
            "max_seconds": data.get("max_seconds", 60),
            "max_bytes": data.get("max_bytes", 256 * 1024**2),
        },
        output_dir / "gp-root-reference",
    )
    if result.timed_out:
        raise ValueError("GP reference timed out; saved dataset unchanged")
    if result.returncode or result.summary is None:
        detail = result.stderr.strip().splitlines()
        raise ValueError("GP reference failed: " + (detail[-1] if detail else "failed"))
    return {
        **result.summary,
        "kind": "gp_root_causal_reference",
        "feature": feature,
        "anchor_row": row,
        "episode_id": task.episode_id,
        "bounds": [low, high],
        "feature_role": schema.kind,
        "score_semantics": (
            "Original-row hidden GP roots and retained independent fit; "
            "explicit observed cuts. Not a conditional expectation."
        ),
    }
