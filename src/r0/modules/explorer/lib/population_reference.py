"""Guarded active/before/after original-row population reference controls."""

import json
import math
from dataclasses import asdict
from typing import TYPE_CHECKING, Any

from r0.modules.explorer.lib.construction import caller_limits
from r0.modules.generation import run_guarded_document

if TYPE_CHECKING:
    from r0.modules.explorer.lib.session import ExplorerSession

POPULATION_REFERENCE_VERSIONS = (
    "calibrated-population-graph-v1",
    "calibrated-population-query-v1",
    "calibrated-population-intervention-v1",
    "recursive-population-graph-v1",
    "recursive-population-graph-intervention-v1",
)


def population_reference_action(session: "ExplorerSession", data: dict[str, Any]) -> dict[str, Any]:
    if (
        session.task is None
        or data.get("expected_episode_id", session.task.episode_id) != session.task.episode_id
    ):
        raise ValueError("population reference active dataset identity is missing or stale")
    side = data.get("population_reference_side", "active")
    if side not in ("active", "before", "after"):
        raise ValueError("population reference requires active, before or after")
    task = session.task
    if side != "active":
        if (
            session.intervention is None
            or data.get("expected_intervention_id") != session.intervention.after.episode_id
        ):
            raise ValueError("population reference comparison identity is missing or stale")
        task = session.intervention.before if side == "before" else session.intervention.after
    record = json.loads(task.recipe_json)
    if record["recipe"].get("version") not in POPULATION_REFERENCE_VERSIONS:
        raise ValueError("selected reference source is not a retained population DAG")
    feature, row, points = data.get("feature"), data.get("anchor_row", 0), data.get("points", 17)
    if (
        type(feature) is not int
        or not 0 <= feature < task.inputs.shape[1]
        or type(row) is not int
        or not 0 <= row < len(task.inputs)
        or type(points) is not int
        or not 2 <= points <= 129
    ):
        raise ValueError("population reference requires bounded feature, original row and grid")
    schema = task.feature_schema[feature] if task.feature_schema else None
    vocabulary = schema.vocabulary if schema else ()
    bounds = data.get("bounds")
    if vocabulary:
        if bounds is not None:
            raise ValueError(
                "categorical reference enumerates declared categories; numeric bounds are unused"
            )
        values = [float(i) for i in range(len(vocabulary))]
        bounds = [0.0, float(len(vocabulary) - 1)]
    else:
        bounds = (
            [float(task.inputs[:, feature].min()), float(task.inputs[:, feature].max())]
            if bounds is None
            else bounds
        )
        if (
            type(bounds) not in (tuple, list)
            or len(bounds) != 2
            or any(type(v) not in (int, float) or not math.isfinite(v) for v in bounds)
            or bounds[0] > bounds[1]
        ):
            raise ValueError("population reference bounds must be finite and ordered")
        values = [
            (1 - i / (points - 1)) * bounds[0] + i / (points - 1) * bounds[1] for i in range(points)
        ]
    result = run_guarded_document(
        {
            "validation": "population-curve",
            "record": record,
            "row": row,
            "feature": feature,
            "values": values,
            "capture_location": data.get("capture_location", "capture"),
            "causal_limits": asdict(caller_limits(data)),
            "max_seconds": data.get("max_seconds", 60),
            "max_bytes": data.get("max_bytes", 256 * 1024**2),
        },
        session.output_dir / "population-reference",
    )
    if result.timed_out:
        raise ValueError("Population reference timed out; saved baseline and comparison unchanged")
    if result.returncode or result.summary is None:
        detail = result.stderr.strip().splitlines()
        raise ValueError(
            "Population reference failed: " + (detail[-1] if detail else "worker failed")
        )
    return {
        "curve": {
            **result.summary,
            "kind": "population_original_row_reference",
            "bounds": bounds,
            "points": len(values),
            "anchor_row": row,
            "source_side": side,
            "feature_role": "declared category" if vocabulary else "observed causal readout",
            "score_semantics": (
                "Original-row forest/mechanism response with fixed keyed draws. "
                "Numeric state/readout and native emitted-variable clamps; "
                "converter location explicit. "
                "Scores/probabilities and realized targets use saved target semantics, "
                "including the retained independent calibration when present; "
                "not a conditional mean. Lines only connect queried values."
            ),
        }
    }
