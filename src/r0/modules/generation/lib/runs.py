"""Single-coordinator resumable runs. Platform entrypoints own hard execution limits."""

import fcntl
import json
import math
import os
import time
import zipfile
from collections.abc import Callable
from contextlib import closing
from dataclasses import asdict, dataclass
from pathlib import Path

from r0.modules.generation.lib.groups import (
    EmptyGroupError,
    GroupLimitError,
    save_group,
    verify_group,
    write_record,
)
from r0.modules.generation.lib.parallel import parallel_tasks
from r0.modules.priors import Recipe, TrainingTask, generate_task


@dataclass(frozen=True, slots=True)
class RunRequest:
    recipe: Recipe
    seed: int
    count: int
    max_seconds: float
    max_bytes: int
    start: int = 0
    group_size: int = 64
    device: str = "cpu"

    def __post_init__(self) -> None:
        if any(
            type(v) is not int
            for v in (self.seed, self.count, self.start, self.group_size, self.max_bytes)
        ):
            raise ValueError("run counts, identities and storage limits must be integers")
        if not (0 <= self.seed < 2**63 and 0 <= self.start < self.start + self.count <= 2**63):
            raise ValueError("invalid seed or task interval")
        if not (1 <= self.count <= 100_000_000 and 1 <= self.group_size <= 1024):
            raise ValueError("count must be 1..100M and group_size 1..1024")
        if self.max_bytes < 65_536 or self.device not in ("cpu", "cuda"):
            raise ValueError("at least 64 KiB storage and an explicit CPU/CUDA device are required")
        if type(self.max_seconds) not in (int, float) or not (
            math.isfinite(self.max_seconds) and self.max_seconds > 0
        ):
            raise ValueError("max_seconds must be finite and positive")


@dataclass(frozen=True, slots=True)
class RunSummary:
    attempted: int
    completed: int
    failed: int
    pending: int
    status: str


def _size(root: Path) -> int:
    return sum(p.stat().st_size for p in root.rglob("*") if p.is_file())


def _previous(group: Path, config: dict[str, object]) -> tuple[set[int], set[int]]:
    completed: set[int] = set()
    failed: set[int] = set()
    for record in sorted(group.glob("attempt-*.json")):
        data = json.loads(record.read_text())
        failed.update(int(i) for i in data["failed"])
    for record in group.glob("failures-*.jsonl"):
        with record.open() as handle:
            for line in handle:
                failed.add(int(json.loads(line)["task_id"]))
    # The archive receipt is the completion oracle, including a saved archive whose
    # process was interrupted before writing its separate attempt summary.
    for archive in sorted(group.glob("data-*.zip")):
        marker = archive.with_suffix(".json")
        # Archive receipts have their own stem to avoid collision with attempt summaries.
        if marker.exists():
            receipt = verify_group(archive)
            with zipfile.ZipFile(archive) as data:
                for task_id in receipt.task_ids:
                    record = json.loads(data.read(f"{task_id}/card.json"))["recipe"]
                    if (
                        record["recipe"] != config["recipe"]
                        or record["seed"] != config["seed"]
                        or record["task_id"] != task_id
                    ):
                        raise ValueError("saved group belongs to a different recipe or task")
            completed.update(receipt.task_ids)
    failed.difference_update(completed)
    return completed, failed


def _record_failure(path: Path, task_id: int, error: Exception) -> str:
    message = f"{type(error).__name__}: {error}"
    # Bound encoded JSON bytes, including escaping, rather than character count.
    while len(json.dumps(message).encode()) > 1024:
        message = message[: len(message) // 2] + "... [truncated]"
    with path.open("a") as handle:
        handle.write(json.dumps({"task_id": task_id, "error": message}) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    return message


def run_generation(
    request: RunRequest,
    out: Path,
    *,
    retry_failed: bool = False,
    produce: Callable[[int], TrainingTask] | None = None,
    workers: int = 1,
    persist: Callable[[], None] | None = None,
) -> RunSummary:
    """Resume verified output, preserving all attempts and requiring explicit retries.

    Time is an admission deadline checked between tasks, not a process kill limit.
    A platform watchdog must enforce a hard wall limit around this coordinator.
    Only one coordinator may write a directory. Provider flush/commit is the caller's
    responsibility; this routine verifies and fsyncs the locally mounted files.
    """
    if type(workers) is not int or not 1 <= workers <= 16:
        raise ValueError("workers must be an integer in 1..16")
    out.mkdir(parents=True, exist_ok=True)
    with (out / ".writer.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise ValueError("another coordinator is writing this run") from error
        return _run(
            request,
            out,
            retry_failed=retry_failed,
            produce=produce,
            workers=workers,
            persist=persist,
        )


def _run(
    request: RunRequest,
    out: Path,
    *,
    retry_failed: bool,
    produce: Callable[[int], TrainingTask] | None,
    workers: int,
    persist: Callable[[], None] | None,
) -> RunSummary:
    def generate(task_id: int) -> TrainingTask:
        return generate_task(
            request.recipe, seed=request.seed, task_id=task_id, device=request.device
        )

    factory = produce or generate
    config = json.loads(json.dumps(asdict(request), sort_keys=True))
    config["format"] = "generation-run-v1"
    header = out / "run.json"
    if header.exists():
        if json.loads(header.read_text()) != config:
            raise ValueError("run configuration changed; choose a new output directory")
    else:
        write_record(header, config)
    deadline = time.monotonic() + request.max_seconds
    used_bytes = _size(out)
    attempted = completed_count = failed_count = 0
    status = "COMPLETE"
    for start in range(request.start, request.start + request.count, request.group_size):
        group = out / "groups" / str(start // (1024 * request.group_size)) / str(start)
        completed, failed = _previous(group, config)
        stop = min(start + request.group_size, request.start + request.count)
        ids = set(range(start, stop))
        if not completed.issubset(ids) or not failed.issubset(ids):
            raise ValueError("saved group task IDs do not match requested interval")
        pending = sorted(ids - completed - (set() if retry_failed else failed))
        if pending and status == "COMPLETE":
            remaining = request.max_bytes - used_bytes
            # Reserve bounded attempt metadata before writing any more archive data.
            reserve = 4096 + 4096 * len(pending)
            if remaining < reserve + 4096:
                status = "STORAGE_LIMIT"
            elif time.monotonic() >= deadline:
                status = "TIME_LIMIT"
            else:
                group.mkdir(parents=True, exist_ok=True)
                previous_bytes = _size(group)
                attempt = len(list(group.glob("start-*.json")))
                write_record(group / f"start-{attempt}.json", {"task_ids": pending})
                errors: dict[str, str] = {}
                yielded: list[int] = []

                def observe(
                    task_id: int,
                    outcome: TrainingTask | Exception,
                    failures: dict[str, str] = errors,
                    emitted: list[int] = yielded,
                    failure_log: Path = group / f"failures-{attempt}.jsonl",
                ) -> TrainingTask | None:
                    nonlocal attempted
                    attempted += 1
                    try:
                        if isinstance(outcome, Exception):
                            raise outcome
                        task = outcome
                        record = json.loads(task.recipe_json)
                        if (
                            record["recipe"] != config["recipe"]
                            or record["seed"] != request.seed
                            or record["task_id"] != task_id
                            or task.inputs.device.type != request.device
                        ):
                            raise ValueError("producer returned a different task or recipe")
                    except (ValueError, RuntimeError) as error:
                        failures[str(task_id)] = _record_failure(failure_log, task_id, error)
                        if persist is not None:
                            persist()
                        return None
                    emitted.append(task_id)
                    return task

                archive = group / f"data-{attempt}.zip"
                saved: tuple[int, ...] = ()
                try:
                    with closing(
                        parallel_tasks(
                            pending, factory, workers=workers, deadline=deadline, observe=observe
                        )
                    ) as tasks:
                        receipt = save_group(archive, tasks, max_bytes=remaining - reserve)
                        saved = receipt.task_ids
                except GroupLimitError:
                    status = "STORAGE_LIMIT"
                except EmptyGroupError:
                    pass
                if status == "COMPLETE" and time.monotonic() > deadline:
                    status = "TIME_LIMIT"
                write_record(
                    group / f"attempt-{attempt}.json",
                    {
                        "completed": saved,
                        "failed": errors,
                        "uncommitted": sorted(set(yielded) - set(saved)),
                        "status": status,
                    },
                )
                if persist is not None:
                    persist()
                completed.update(saved)
                failed.update(int(i) for i in errors)
                failed.difference_update(completed)
                used_bytes += _size(group) - previous_bytes
        completed_count += len(completed)
        failed_count += len(failed)
    pending_count = request.count - completed_count - failed_count
    if time.monotonic() > deadline and status == "COMPLETE":
        status = "TIME_LIMIT"
    if status == "COMPLETE" and failed_count:
        status = "FAILED_TASKS"
    return RunSummary(attempted, completed_count, failed_count, pending_count, status)
