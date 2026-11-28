"""Read-only accounting of locally persisted coverage shards."""

import fcntl
import json
import zipfile
from dataclasses import asdict
from pathlib import Path
from typing import Any

from r0.modules.generation.lib.coverage import CoveragePlan, coverage_shard
from r0.modules.generation.lib.groups import verify_group
from r0.modules.generation.lib.requests import record_matches, request_from_dict


def inspect_coverage_shard(plan: CoveragePlan, index: int, out: Path) -> dict[str, Any]:
    """Verify committed archives and account for recorded admissions/failures.

    Reads local runner output, not arbitrary uploaded archives. Refuses inspection
    during a shard write. Admission is not proof a task actually ran. Checksums
    and request binding are not numerical replay or measured mechanism diversity.
    """
    shard = coverage_shard(plan, index)
    manifest = out / "coverage-plan.json"
    if manifest.exists() and (
        manifest.stat().st_size > 8 * 1024**2
        or json.loads(manifest.read_text()) != json.loads(plan.to_json())
    ):
        raise ValueError("coverage output belongs to a different plan")
    root = out / "shards" / f"{index:08d}"
    if not root.exists():
        return _report(plan, index, set(), set(), set(), 0, 0, 0, 0)
    if not manifest.exists():
        raise ValueError("shard output has no coverage manifest")
    lock_path = root / ".writer.lock"
    if not lock_path.exists():
        raise ValueError("shard output has no coordinator lock")
    with lock_path.open("r") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_SH | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise ValueError("cannot inspect an actively written shard") from error
        header = root / "run.json"
        if not header.exists():
            raise ValueError("shard has no run header; initialization may be interrupted")
        document = json.loads(header.read_text())
        version = document.pop("format", None)
        expected_version = (
            "generation-run-v1" if shard.request.sampling is None else "generation-run-v2"
        )
        if version != expected_version or asdict(request_from_dict(document)) != asdict(
            shard.request
        ):
            raise ValueError("shard run header differs from coverage plan")
        admitted: set[int] = set()
        failed: set[int] = set()
        committed: set[int] = set()
        admissions = failure_events = archive_bytes = incomplete = 0

        def valid(task_id: Any) -> int:
            if (
                type(task_id) is not int
                or not shard.request.start <= task_id < shard.request.start + shard.request.count
            ):
                raise ValueError("recorded task identity lies outside coverage shard")
            return task_id

        for path in root.glob("groups/*/*/start-*.json"):
            for task_id in json.loads(path.read_text())["task_ids"]:
                admitted.add(valid(task_id))
                admissions += 1
        for path in root.glob("groups/*/*/failures-*.jsonl"):
            with path.open() as handle:
                for line in handle:
                    failed.add(valid(json.loads(line)["task_id"]))
                    failure_events += 1
        # Completion summaries can survive without a failure log (older runs).
        # They supplement failed IDs, but do not invent additional failure events.
        for path in root.glob("groups/*/*/attempt-*.json"):
            for task_id in json.loads(path.read_text())["failed"]:
                failed.add(valid(int(task_id)))
        for path in root.glob("groups/*/*/data-*.zip"):
            if not path.with_suffix(".json").exists():
                incomplete += 1
                continue
            receipt = verify_group(path)
            with zipfile.ZipFile(path) as archive:
                for task_id in receipt.task_ids:
                    valid(task_id)
                    if task_id in committed:
                        raise ValueError("duplicate committed task identity across archives")
                    record = json.loads(archive.read(f"{task_id}/card.json"))["recipe"]
                    if not record_matches(record, shard.request, task_id):
                        raise ValueError("saved task belongs to a different request")
                    committed.add(task_id)
            archive_bytes += receipt.size_bytes
        return _report(
            plan,
            index,
            admitted,
            failed,
            committed,
            admissions,
            failure_events,
            archive_bytes,
            incomplete,
        )


def _report(
    plan: CoveragePlan,
    index: int,
    admitted: set[int],
    failed: set[int],
    committed: set[int],
    admissions: int,
    failure_events: int,
    archive_bytes: int,
    incomplete: int,
) -> dict[str, Any]:
    shard = coverage_shard(plan, index)
    unresolved = failed - committed
    return {
        "plan_id": plan.identity,
        "shard": index,
        "stratum": shard.stratum,
        "requested": shard.request.count,
        "committed": len(committed),
        "failed_unresolved": len(unresolved),
        "failed_then_committed": len(failed & committed),
        "pending": shard.request.count - len(committed) - len(unresolved),
        "admitted_unique": len(admitted),
        "admission_events": admissions,
        "recorded_failure_events": failure_events,
        "admitted_without_outcome": len(admitted - failed - committed),
        "verified_archive_bytes": archive_bytes,
        "incomplete_archives": incomplete,
        "status": "COMPLETE" if len(committed) == shard.request.count else "INCOMPLETE",
        "evidence": "archive integrity and request binding; numerical replay not performed",
        "mechanism_diversity": "NOT_MEASURED",
    }


def inspect_coverage(plan: CoveragePlan, out: Path) -> dict[str, Any]:
    """Stream existing shards into stratum totals; memory does not grow per task.

    Each shard is locked during inspection. This is not an atomic campaign-wide
    snapshot: stop all writers for a final acceptance report.
    """
    from r0.modules.generation.lib.coverage import coverage_allocations

    manifest = out / "coverage-plan.json"
    if manifest.exists() and (
        manifest.stat().st_size > 8 * 1024**2
        or json.loads(manifest.read_text()) != json.loads(plan.to_json())
    ):
        raise ValueError("coverage output belongs to a different plan")
    fields = (
        "committed",
        "failed_unresolved",
        "failed_then_committed",
        "admitted_unique",
        "admission_events",
        "recorded_failure_events",
        "admitted_without_outcome",
        "verified_archive_bytes",
        "incomplete_archives",
    )
    strata = {
        a.key: {"requested": a.count, **dict.fromkeys(fields, 0)}
        for a in coverage_allocations(plan)
    }
    root = out / "shards"
    inspected = 0
    if root.exists():
        for directory in root.iterdir():
            if not directory.is_dir() or not directory.name.isdigit():
                raise ValueError("unexpected entry in coverage shard directory")
            index = int(directory.name)
            if directory.name != f"{index:08d}":
                raise ValueError("noncanonical coverage shard directory")
            report = inspect_coverage_shard(plan, index, out)
            bucket = strata[report["stratum"]]
            for field in fields:
                bucket[field] += report[field]
            inspected += 1
    marginals: dict[str, dict[str, int]] = {}
    for stratum in plan.strata:
        bucket = strata[stratum.key]
        bucket["pending"] = bucket["requested"] - bucket["committed"] - bucket["failed_unresolved"]
        for dimension, value in stratum.tags:
            marginal = marginals.setdefault(dimension, {})
            marginal[value] = marginal.get(value, 0) + bucket["committed"]
    totals = {
        field: sum(bucket[field] for bucket in strata.values())
        for field in ("requested", "pending", *fields)
    }
    return {
        "plan_id": plan.identity,
        "status": "COMPLETE" if totals["committed"] == plan.total else "INCOMPLETE",
        "snapshot": "per-shard locked; stop writers for a final campaign snapshot",
        "inspected_shards": inspected,
        "totals": totals,
        "strata": strata,
        "committed_tag_marginals": marginals,
        "tag_basis": "caller-declared strata, not measured task properties",
        "mechanism_diversity": "NOT_MEASURED",
        "evidence": "archive integrity and request binding; numerical replay not performed",
    }
