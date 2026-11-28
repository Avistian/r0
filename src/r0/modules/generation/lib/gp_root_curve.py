"""Killable original-row GP-root reference grids with exact receipt downloads."""

import json
import math
from dataclasses import asdict
from typing import Any, cast

from r0.modules import priors as p


def gp_root_curve(data: dict[str, Any]) -> dict[str, Any]:
    maximum = data.get("max_bytes", 256 * 1024**2)
    if type(maximum) is not int or not 1 <= maximum <= 256 * 1024**2:
        raise ValueError("GP reference byte cap must be 1..268435456")
    request_bytes = len(json.dumps(data, allow_nan=False).encode())
    if request_bytes > maximum:
        raise ValueError("GP reference request exceeds byte cap")
    try:
        source = data["recipe"]
        current = source if source["version"] == "gp-root-dag-v1" else source["original"]
        if any(
            type(data.get(key)) is not int or data[key] != current["source"][key]
            for key in ("seed", "task_id")
        ):
            raise ValueError("GP reference retained identity differs")
        limits = p.CausalLimits(**data["causal_limits"])
        features = current["base"]["task"]["features"]
        if type(features) is not int or not 1 <= features <= 256:
            raise ValueError("GP reference feature count is outside task geometry")
        feature = data["feature"]
        anchor = data["anchor"]
        if (
            type(anchor) is not list
            or type(feature) is not int
            or len(cast(list[Any], anchor)) != features
            or not 0 <= feature < features
        ):
            raise ValueError("GP reference feature/anchor geometry differs")
        anchor = cast(list[Any], anchor)
        grid: Any = data.get("inputs")
        count = len(cast(list[Any], grid)) if type(grid) is list else len(data.get("values", []))
        if (
            type(count) is not int
            or not 1 <= count <= 129
            or count * features > limits.parameters
            or 8 * request_bytes + 256 * count * features > limits.working_bytes
        ):
            raise ValueError("GP reference grid metadata exceeds caller limits")
        if grid is None:
            values = data["values"]
            if type(values) is not list or not 1 <= len(cast(list[Any], values)) <= 129:
                raise ValueError("GP reference grid needs 1..129 points")
            values = cast(list[Any], values)
            grid = [anchor[:feature] + [value] + anchor[feature + 1 :] for value in values]
        if type(cast(Any, grid)) is not list or not 1 <= len(cast(list[Any], grid)) <= 129:
            raise ValueError("GP reference grid needs 1..129 rows")
        grid = cast(list[Any], grid)
        if any(
            type(row) is not list
            or len(row) != len(anchor)
            or any(
                type(v) not in (int, float) or not math.isfinite(v) for v in cast(list[Any], row)
            )
            for row in grid
        ):
            raise ValueError("GP reference grid requires finite numeric rows")
        result = p.gp_root_reference_from_source(
            source,
            row=data["anchor_row"],
            inputs=tuple(tuple(float(v) for v in row) for row in grid),
            structural_innovations=data.get("reference", "original-row"),
            capture_location=data.get("capture_location", "capture"),
            limits=limits,
        )
    except (KeyError, TypeError, OverflowError) as error:
        raise ValueError("malformed GP reference request") from error
    raw = asdict(result)
    # Preserve every integer in downloadable JSON; browser-facing IDs use strings.
    response = {
        **raw,
        "reference_json": json.dumps(raw, sort_keys=True, separators=(",", ":"), allow_nan=False),
        "seed": str(result.seed),
        "task_id": str(result.task_id),
        "values": [row[feature] for row in result.inputs],
        "points": len(result.inputs),
    }
    if len(json.dumps(response, allow_nan=False).encode()) > maximum:
        raise ValueError("GP reference response exceeds byte cap")
    return response
