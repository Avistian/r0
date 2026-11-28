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
    record = cast(dict[str, Any], data["record"])
    if type(record) is not dict or set(record) != {"recipe", "seed", "task_id", "execution"}:
        raise ValueError("population reference requires complete saved source identity")
    source = record["recipe"]
    base = source.get("base", source)
    if any(type(record[k]) is not int or record[k] != base[k] for k in ("seed", "task_id")):
        raise ValueError("population reference transport identity differs")
    execution = cast(dict[str, Any], record["execution"])
    execution_keys = {
        "implementation",
        "device",
        "python",
        "numpy",
        "torch",
        "cpu_threads",
        "cuda",
        "gpu",
    }
    if (
        type(execution) is not dict
        or set(execution) != execution_keys
        or any(type(execution[k]) is not str for k in ("python", "numpy", "torch"))
        or type(execution["cpu_threads"]) is not int
        or execution["cpu_threads"] < 1
        or any(execution[k] is not None and type(execution[k]) is not str for k in ("cuda", "gpu"))
        or execution.get("implementation") != source.get("version")
        or execution.get("device") != "cpu"
    ):
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
