"""Prepare fixed episodes once; no predictor owns sampling or scoring targets."""

import math
from collections import Counter
from pathlib import Path
from typing import Any

from .storage import Document, digest, file_hash, read_json, seal, write_json

SAMPLER = "sha256-json-seed-row-v1; squared-deviation-greedy; declared-class-ties"
LIMITS: Document = {
    "cpu_max": "100000 100000",
    "threads": 1,
    "memory_max": 8 * 1024**3,
    "swap_max": 0,
    "wall_seconds": 300,
}


def row_identity(source_hash: str, parser: str, position: int) -> str:
    if position < 0:
        raise ValueError("negative source position")
    return digest(["row-v1", source_hash, parser, position])


def _rank(row: Document, seed: int) -> tuple[str, str]:
    return digest(["rank-v1", seed, row["id"]]), row["id"]


def _select(
    rows: list[Document],
    count: int,
    seed: int,
    classes: list[str],
    proportions: list[float],
    reserve: int = 0,
) -> list[Document]:
    if not classes:
        if len(rows) < count + reserve:
            raise ValueError("infeasible support")
        return sorted(rows, key=lambda r: _rank(r, seed))[:count]
    groups = [[r for r in rows if r["target"] == c] for c in classes]
    capacities = [len(g) - reserve for g in groups]
    if count < len(classes) or min(capacities) < 1 or sum(capacities) < count:
        raise ValueError("infeasible class support; no seed search")
    allocated = [1] * len(classes)
    for _ in range(count - len(classes)):
        choices = [i for i in range(len(classes)) if allocated[i] < capacities[i]]
        selected = min(choices, key=lambda i: (2 * (allocated[i] - count * proportions[i]) + 1, i))
        allocated[selected] += 1
    chosen = [
        r
        for g, n in zip(groups, allocated, strict=True)
        for r in sorted(g, key=lambda r: _rank(r, seed))[:n]
    ]
    return sorted(chosen, key=lambda r: _rank(r, seed))


def prepare(
    source: Path, destination: Path, *, query_count: int = 4, context_count: int = 8
) -> Document:
    """Read an authored JSON fixture; refuse real sources until operational admission exists.

    Row identities precede exclusions and bind file bytes and parser recipe. A frozen
    receipt alone is intentionally insufficient to authorize a real execution lane.
    """
    data = read_json(source)
    if data.get("evidence") != "AUTHORED_FIXTURE":
        raise ValueError(
            "real execution blocked: admission/freeze/access/budget/learner receipts PENDING"
        )
    if query_count < 1 or context_count < 1:
        raise ValueError("positive support required")
    task = data["task"]
    classes: list[str] = data.get("classes", [])
    if task not in ("regression", "binary", "multiclass"):
        raise ValueError("unsupported task")
    if (
        (task == "regression" and classes)
        or (task == "binary" and len(classes) != 2)
        or (task == "multiclass" and len(classes) < 3)
        or len(set(classes)) != len(classes)
    ):
        raise ValueError("invalid declared class order")
    schema: list[Document] = data["features"]
    if (
        not schema
        or len({f["name"] for f in schema}) != len(schema)
        or any(f["kind"] not in ("numeric", "categorical") for f in schema)
    ):
        raise ValueError("invalid schema (temporal datasets are not admitted)")
    parser = "authored-json-rows-v1"
    source_hash = file_hash(source)
    rows: list[Document] = []
    exclusions: list[Document] = []
    row_metadata: list[Document] = []
    seen: set[str] = set()
    for position, raw in enumerate(data["rows"]):
        row: Document = {
            **raw,
            "id": row_identity(source_hash, parser, position),
            "position": position,
        }
        row_metadata.append(
            {"id": row["id"], "position": position, "source_id": raw.get("source_id")}
        )
        values: list[Any] = row["values"]
        if len(values) != len(schema):
            raise ValueError("feature width mismatch")
        for feature, value in zip(schema, values, strict=True):
            if value is None:
                continue
            if feature["kind"] == "categorical":
                if not isinstance(value, str):
                    raise ValueError("categories must be strings")
            elif type(value) not in (int, float) or not math.isfinite(value):
                raise ValueError("numeric features must be finite or null")
        target = row.get("target")
        valid = (
            (
                isinstance(target, (int, float))
                and not isinstance(target, bool)
                and math.isfinite(target)
            )
            if task == "regression"
            else target in classes
        )
        key = digest([values, target]) if valid else ""
        if not valid or key in seen:
            exclusions.append(
                {
                    "id": row["id"],
                    "position": position,
                    "reason": "duplicate" if valid else "invalid_target",
                }
            )
        else:
            seen.add(key)
            rows.append(row)
    labels: dict[str, set[str]] = {}
    for row in rows:
        labels.setdefault(digest(row["values"]), set()).add(digest(row["target"]))
    eligible: list[Document] = []
    for row in rows:
        if len(labels[digest(row["values"])]) > 1:
            exclusions.append({"id": row["id"], "position": row["position"], "reason": "conflict"})
        else:
            eligible.append(row)
    counts = Counter(r["target"] for r in eligible)
    proportions = [counts[c] / len(eligible) for c in classes] if eligible else []
    if len(eligible) < query_count + context_count:
        raise ValueError("infeasible disjoint query/context support")
    queries = _select(
        eligible, query_count, 0, classes, proportions, reserve=1 if classes else context_count
    )
    query_ids = [r["id"] for r in queries]
    pool = [r for r in eligible if r["id"] not in query_ids]
    contexts = [_select(pool, context_count, seed, classes, proportions) for seed in (1, 2, 3)]
    destination.mkdir(parents=True, exist_ok=False)
    inputs = destination / "inputs"
    inputs.mkdir()
    draws: list[Document] = []
    for seed, context in enumerate(contexts, 1):
        view = {
            "version": 1,
            "evidence": "AUTHORED_FIXTURE",
            "task": task,
            "classes": classes,
            "features": schema,
            "context_ids": [r["id"] for r in context],
            "query_ids": query_ids,
            "context_values": [r["values"] for r in context],
            "query_values": [r["values"] for r in queries],
            "context_targets": [r["target"] for r in context],
            "support": context_count,
            "limits": LIMITS,
        }
        path = inputs / f"draw-{seed}.json"
        write_json(path, view)
        draws.append(
            {
                "seed": seed,
                "input": str(path.relative_to(destination)),
                "sha256": file_hash(path),
                "context_ids": view["context_ids"],
            }
        )
    write_json(
        destination / "scoring.json",
        {"version": 1, "query_ids": query_ids, "targets": [r["target"] for r in queries]},
    )
    manifest = {
        "version": 1,
        "evidence": "AUTHORED_FIXTURE",
        "protocol": "fixture-v1-not-frozen",
        "dataset": data["dataset"],
        "partition": data.get("partition", "development"),
        "task": task,
        "classes": classes,
        "features": schema,
        "source_sha256": source_hash,
        "parser": parser,
        "sampler": SAMPLER,
        "query_seed": 0,
        "query_ids": query_ids,
        "query_count": query_count,
        "context_count": context_count,
        "draws": draws,
        "exclusions": exclusions,
        "row_metadata": row_metadata,
        "eligible_count": len(eligible),
        "limits": LIMITS,
        "scoring_sha256": file_hash(destination / "scoring.json"),
    }
    write_json(destination / "manifest.json", manifest)
    seal(destination)
    return manifest
