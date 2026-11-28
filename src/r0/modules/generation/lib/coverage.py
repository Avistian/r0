"""Exact finite allocations and lazy shards for reproducible dataset portfolios."""

import fcntl
import hashlib
import json
import re
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, cast

from r0.modules.generation.lib.coverage_hierarchy import (
    CoverageHierarchy,
    hierarchical_counts,
    hierarchy_from_dict,
    quota_counts,
)
from r0.modules.generation.lib.groups import write_record
from r0.modules.generation.lib.job import JobResult, run_guarded
from r0.modules.generation.lib.requests import RunRequest, request_from_dict

_NAME = re.compile(r"[a-zA-Z0-9][a-zA-Z0-9_.-]{0,95}\Z")
_MAX_DOCUMENT = 8 * 1024**2


def _canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _fresh(request: RunRequest) -> bool:
    if request.sampling is not None:
        return True
    if request.construction is not None:
        # Other construction adapters can retain the same construction seed for
        # every episode. Do not infer independent mechanisms from unique task IDs.
        if json.loads(request.construction.controls_json).get("topology") == "per-task":
            from r0.modules.generation.lib.task_construction import task_construction_source

            task_construction_source(request.construction)
            return True
        return json.loads(request.construction.controls_json).get("topology") in (
            "tree-extension",
            "tree-ensemble",
        )
    return False


@dataclass(frozen=True, slots=True)
class CoverageStratum:
    key: str
    request: RunRequest
    weight: int = 1
    minimum: int = 1
    tags: tuple[tuple[str, str], ...] = ()

    def __post_init__(self) -> None:
        if (
            type(self.key) is not str
            or not _NAME.fullmatch(self.key)
            or type(self.request) is not RunRequest
            or self.request.count != 1
            or self.request.start != 0
            or type(self.weight) is not int
            or not 0 <= self.weight <= 10**12
            or type(self.minimum) is not int
            or not 0 <= self.minimum <= 100_000_000
            or type(self.tags) is not tuple
            or len(self.tags) > 16
        ):
            raise ValueError(
                "stratum requires a named one-task template and bounded integer allocation"
            )
        if any(
            type(pair) is not tuple
            or len(pair) != 2
            or any(type(v) is not str or not _NAME.fullmatch(v) for v in pair)
            for pair in self.tags
        ) or len({k for k, _ in self.tags}) != len(self.tags):
            raise ValueError("coverage tags require unique bounded dimension names")


@dataclass(frozen=True, slots=True)
class CoveragePlan:
    strata: tuple[CoverageStratum, ...]
    total: int
    shard_size: int = 1000
    start: int = 0
    require_fresh: bool = True
    version: str = "coverage-plan-v1"
    hierarchy: CoverageHierarchy | None = None

    def __post_init__(self) -> None:
        if (
            self.version not in ("coverage-plan-v1", "coverage-plan-v2")
            or (self.hierarchy is None) != (self.version == "coverage-plan-v1")
            or (self.hierarchy is not None and type(self.hierarchy) is not CoverageHierarchy)
            or type(self.strata) is not tuple
            or not 1 <= len(self.strata) <= 4096
            or any(type(s) is not CoverageStratum for s in self.strata)
            or len({s.key for s in self.strata}) != len(self.strata)
            or type(self.total) is not int
            or not 1 <= self.total <= 100_000_000
            or type(self.start) is not int
            or not 0 <= self.start < self.start + self.total <= 2**63
            or type(self.shard_size) is not int
            or not 1 <= self.shard_size <= 1_000_000
            or type(self.require_fresh) is not bool
        ):
            raise ValueError("invalid bounded coverage plan")
        minimum = sum(s.minimum for s in self.strata)
        if minimum > self.total or (
            minimum < self.total and not any(s.weight for s in self.strata)
        ):
            raise ValueError("coverage minimums exceed total or residual allocation has no weight")
        dimensions = {k for k, _ in self.strata[0].tags}
        if any({k for k, _ in s.tags} != dimensions for s in self.strata):
            raise ValueError("every stratum must declare the same coverage dimensions")
        if self.require_fresh and any(not _fresh(s.request) for s in self.strata):
            raise ValueError(
                "independent-source plan requires per-task sampling or sampled tree construction"
            )
        coverage_allocations(self)
        if len(self.to_json().encode()) > _MAX_DOCUMENT:
            raise ValueError("coverage plan document exceeds 8 MiB")

    def to_json(self) -> str:
        data = asdict(self)
        if self.hierarchy is None:
            data.pop("hierarchy")
        else:
            data["hierarchy"]["weights"] = sorted(data["hierarchy"]["weights"])
        data["strata"] = sorted(data["strata"], key=lambda s: s["key"])
        for stratum in data["strata"]:
            observation = stratum["request"].get("observation")
            if observation is not None and not observation.get("quantiles"):
                observation.pop("quantiles", None)
            stratum["tags"] = sorted(stratum["tags"])
        return _canonical(data)

    @property
    def identity(self) -> str:
        return hashlib.sha256(self.to_json().encode()).hexdigest()


@dataclass(frozen=True, slots=True)
class CoverageAllocation:
    key: str
    start: int
    count: int
    first_shard: int
    shards: int


def coverage_allocations(plan: CoveragePlan) -> tuple[CoverageAllocation, ...]:
    """Minimums first, then exact integer largest-remainder allocation.

    Equal remainders break ties by stratum key. Input ordering is immaterial.
    Allocations describe requested attempts, never accepted-output counts.
    """
    if type(plan) is not CoveragePlan:
        raise ValueError("coverage allocation requires a frozen plan")
    ordered = sorted(plan.strata, key=lambda s: s.key)
    arguments = (
        [s.key for s in ordered],
        [s.minimum for s in ordered],
        [s.weight for s in ordered],
        plan.total,
    )
    counts = (
        quota_counts(*arguments)
        if plan.hierarchy is None
        else hierarchical_counts(
            plan.hierarchy,
            arguments[0],
            [dict(s.tags) for s in ordered],
            arguments[1],
            arguments[2],
            arguments[3],
        )
    )
    allocations: list[CoverageAllocation] = []
    offset, shard = plan.start, 0
    for stratum, count in zip(ordered, counts, strict=True):
        shards = (count + plan.shard_size - 1) // plan.shard_size
        allocations.append(CoverageAllocation(stratum.key, offset, count, shard, shards))
        offset += count
        shard += shards
    return tuple(allocations)


@dataclass(frozen=True, slots=True)
class CoverageShard:
    plan_id: str
    index: int
    stratum: str
    request: RunRequest


def coverage_shard(plan: CoveragePlan, index: int) -> CoverageShard:
    if type(index) is not int or index < 0:
        raise ValueError("coverage shard index must be a nonnegative integer")
    for allocation in coverage_allocations(plan):
        if allocation.first_shard <= index < allocation.first_shard + allocation.shards:
            offset = (index - allocation.first_shard) * plan.shard_size
            template = next(s.request for s in plan.strata if s.key == allocation.key)
            request = replace(
                template,
                start=allocation.start + offset,
                count=min(plan.shard_size, allocation.count - offset),
            )
            return CoverageShard(plan.identity, index, allocation.key, request)
    raise ValueError("coverage shard index lies outside plan")


def planned_coverage(
    plan: CoveragePlan, *, interactions: tuple[tuple[str, ...], ...] = ()
) -> dict[str, Any]:
    """Report declared marginal and selected joint quotas, never observed diversity."""
    dimensions = {key for key, _ in plan.strata[0].tags}
    if (
        type(interactions) is not tuple
        or len(interactions) > 64
        or any(
            type(group) is not tuple
            or not 2 <= len(group) <= 4
            or any(type(key) is not str or key not in dimensions for key in group)
            or len(set(group)) != len(group)
            for group in interactions
        )
    ):
        raise ValueError("interactions require up to 64 groups of 2..4 distinct tag dimensions")
    canonical = tuple(sorted({tuple(sorted(group)) for group in interactions}))
    allocations = coverage_allocations(plan)
    counts = {a.key: a.count for a in allocations}
    marginals: dict[str, dict[str, int]] = {}
    for stratum in plan.strata:
        for dimension, value in stratum.tags:
            bucket = marginals.setdefault(dimension, {})
            bucket[value] = bucket.get(value, 0) + counts[stratum.key]
    joints: list[dict[str, Any]] = []
    for group in canonical:
        cells: dict[tuple[str, ...], int] = {}
        for stratum in plan.strata:
            tags = dict(stratum.tags)
            values = tuple(tags[key] for key in group)
            cells[values] = cells.get(values, 0) + counts[stratum.key]
        joints.append(
            {
                "dimensions": list(group),
                "cells": [
                    {"values": list(values), "count": count}
                    for values, count in sorted(cells.items())
                ],
            }
        )
    return {
        "plan_id": plan.identity,
        "status": "PLANNED_NOT_GENERATED",
        "tag_basis": "caller-declared strata, not measured task properties",
        "total": plan.total,
        "shards": sum(a.shards for a in allocations),
        "independent_source_sampling_required": plan.require_fresh,
        "allocations": [asdict(a) for a in allocations],
        "marginals": marginals,
        "interactions": joints,
    }


def coverage_plan_from_json(document: str) -> CoveragePlan:
    if type(document) is not str or len(document.encode()) > _MAX_DOCUMENT:
        raise ValueError("coverage document must be bounded JSON")
    decoded = json.loads(document)
    if type(decoded) is not dict:
        raise ValueError("coverage document must be an object")
    data = cast(dict[str, Any], decoded)
    extra: set[str] = {"hierarchy"} if data.get("version") == "coverage-plan-v2" else set()
    if set(data) != extra | {
        "strata",
        "total",
        "shard_size",
        "start",
        "require_fresh",
        "version",
    }:
        raise ValueError("coverage document fields differ")
    if extra:
        data["hierarchy"] = hierarchy_from_dict(data["hierarchy"])
    raw = data.pop("strata")
    if type(raw) is not list or not 1 <= len(cast(list[Any], raw)) <= 4096:
        raise ValueError("coverage document requires bounded strata")
    strata: list[CoverageStratum] = []
    for value in cast(list[Any], raw):
        if type(value) is not dict:
            raise ValueError("coverage stratum must be an object")
        item = cast(dict[str, Any], value)
        if type(item) is not dict or set(item) != {"key", "request", "weight", "minimum", "tags"}:
            raise ValueError("coverage stratum fields differ")
        tags = item["tags"]
        if type(tags) is not list or any(
            type(p) is not list or len(p) != 2 for p in cast(list[Any], tags)
        ):
            raise ValueError("coverage tags must be pair arrays")
        strata.append(
            CoverageStratum(
                item["key"],
                request_from_dict(item["request"]),
                item["weight"],
                item["minimum"],
                tuple(tuple(p) for p in cast(list[Any], tags)),
            )
        )
    return CoveragePlan(tuple(strata), **data)


def run_coverage_shard(
    plan: CoveragePlan,
    index: int,
    out: Path,
    *,
    workers: int = 1,
    retry_failed: bool = False,
) -> JobResult:
    """Run one requested shard; no hidden loop launches the whole portfolio.

    Existing guarded generation owns time/storage limits and per-shard locking.
    One filesystem coordinator is required per shard on mounts without flock.
    """
    shard = coverage_shard(plan, index)
    out.mkdir(parents=True, exist_ok=True)
    with (out / ".coverage.lock").open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        path = out / "coverage-plan.json"
        document = plan.to_json()
        if path.exists():
            if (
                path.stat().st_size > _MAX_DOCUMENT
                or _canonical(json.loads(path.read_text())) != document
            ):
                raise ValueError("coverage output belongs to a different plan")
        else:
            write_record(path, json.loads(document))
        fcntl.flock(lock, fcntl.LOCK_UN)
    return run_guarded(
        shard.request, out / "shards" / f"{index:08d}", workers=workers, retry_failed=retry_failed
    )
