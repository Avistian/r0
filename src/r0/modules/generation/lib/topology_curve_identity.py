"""Cheap topology reference controls and complete source identity before restoration."""

import hashlib
import json
import math
from typing import Any, cast

from r0.modules import priors


def topology_curve_identity(data: dict[str, Any]) -> dict[str, Any]:
    source = data["recipe"]
    if source.get("version") not in (
        "sampled-hyperprior-graph-v1",
        "sampled-joint-hyperprior-graph-v1",
    ):
        return {}
    try:
        fields = dict(source["request"]["task"])
        fields["class_offsets"] = tuple(fields["class_offsets"])
        task = priors.LinearRecipe(**fields)
        anchor, feature, points, bounds = (
            data["anchor"],
            data["feature"],
            data["points"],
            data["bounds"],
        )
        if type(anchor) is not list or type(bounds) is not list:
            raise ValueError("topology reference requires array controls")
        anchor = cast(list[Any], anchor)
        bounds = cast(list[Any], bounds)
        if (
            len(anchor) != task.features
            or any(type(v) not in (int, float) or not math.isfinite(v) for v in anchor)
            or type(feature) is not int
            or not 0 <= feature < task.features
            or type(points) is not int
            or not 2 <= points <= 512
            or len(bounds) != 2
            or any(type(v) not in (int, float) or not math.isfinite(v) for v in bounds)
            or bounds[0] > bounds[1]
            or any(
                type(data[k]) is not int or not 0 <= data[k] < 2**63 for k in ("seed", "task_id")
            )
            or type(data["anchor_row"]) is not int
            or not 0 <= data["anchor_row"] < task.rows
            or data.get("reference", "original-row") not in ("zero", "original-row")
        ):
            raise ValueError("invalid topology reference controls or original identity")
    except (KeyError, TypeError, AttributeError) as error:
        raise ValueError("invalid topology reference metadata") from error
    source_json = json.dumps(
        {"recipe": source, "seed": data["seed"], "task_id": data["task_id"]},
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    return {
        "source_json": source_json,
        "source_sha256": hashlib.sha256(source_json.encode()).hexdigest(),
        "source_version": source["version"],
        "original_seed": str(data["seed"]),
        "reference_version": "topology-causal-reference-v1",
    }
