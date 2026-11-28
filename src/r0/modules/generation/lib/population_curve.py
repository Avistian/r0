"""Original-row population reference execution stays inside the bounded worker."""

import json
from dataclasses import asdict
from typing import Any, cast

from r0.modules import priors as p


def population_curve(data: dict[str, Any]) -> dict[str, Any]:
    maximum = data.get("max_bytes", 256 * 1024**2)
    if type(maximum) is not int or not 1 <= maximum <= 256 * 1024**2:
        raise ValueError("population reference requires a bounded byte cap")
    if len(json.dumps(data, allow_nan=False).encode()) > maximum:
        raise ValueError("population reference request exceeds byte cap")
    limits = p.CausalLimits(**data["causal_limits"])
    record = p.validate_population_record_metadata(data["record"])
    source = record["recipe"]
    if source.get("version") in (
        "calibrated-population-graph-v1",
        "calibrated-population-query-v1",
        "calibrated-population-intervention-v1",
    ):
        p.preflight_calibrated_generation(source, seed=record["seed"], task_id=record["task_id"])
    else:
        base = source.get("base", source)
        if type(base) is not dict or any(
            record[k] != cast(dict[str, Any], base).get(k) for k in ("seed", "task_id")
        ):
            raise ValueError("population reference transport identity differs")
    execution = record["execution"]
    if execution["implementation"] != source.get("version") or execution["device"] != "cpu":
        raise ValueError("population reference requires its saved CPU implementation identity")
    values = cast(list[Any], data["values"])
    if type(values) is not list:
        raise ValueError("population reference values must be an array")
    result = p.population_reference_from_source(
        source,
        row=data["row"],
        feature=data["feature"],
        values=tuple(values),
        capture_location=data["capture_location"],
        limits=limits,
    )
    raw = asdict(result)
    answer = {
        **raw,
        "reference_json": json.dumps(raw, sort_keys=True, separators=(",", ":")),
        "seed": str(result.seed),
        "task_id": str(result.task_id),
        "population_index": str(result.population_index)
        if result.population_index is not None
        else None,
    }
    if len(json.dumps(answer, allow_nan=False).encode()) > maximum:
        raise ValueError("population reference output exceeds byte cap")
    return answer
