"""Killable finite-GP target references retain original population identities."""

import json
from dataclasses import asdict
from typing import Any

from r0.modules import priors as p


def gp_column_curve(data: dict[str, Any]) -> dict[str, Any]:
    maximum = data.get("max_bytes", 256 * 1024**2)
    if type(maximum) is not int or not 1 <= maximum <= 256 * 1024**2:
        raise ValueError("GP target reference byte cap must be 1..268435456")
    if len(json.dumps(data, allow_nan=False).encode()) > maximum:
        raise ValueError("GP target reference request exceeds byte cap")
    raw = data["recipe"]
    source = raw["source"]
    if any(
        type(data.get(key)) is not int or data[key] != source.get(key)
        for key in ("seed", "task_id")
    ):
        raise ValueError("GP target reference original identity differs")
    bounds = data.get("bounds")
    if bounds is not None and type(bounds) not in (list, tuple):
        raise ValueError("GP target reference bounds require an array")
    result = asdict(
        p.gp_column_reference_from_dict(
            raw,
            feature=data["feature"],
            anchor_row=data["anchor_row"],
            points=data["points"],
            bounds=None if bounds is None else tuple(bounds),
            limits=p.CausalLimits(**data["causal_limits"]),
        )
    )
    # JavaScript receives identities as exact decimal strings, including JSON export.
    result["seed"], result["task_id"] = str(result["seed"]), str(result["task_id"])
    if len(json.dumps(result, allow_nan=False).encode()) > maximum:
        raise ValueError("GP target reference response exceeds byte cap")
    return result
