"""Single-coordinator resumable runs. Platform entrypoints own hard execution limits."""

import fcntl
import json
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
from r0.modules.generation.lib.loading import load_group
from r0.modules.generation.lib.parallel import parallel_tasks
from r0.modules.generation.lib.requests import (
    RunRequest,
    generate_requested,
    record_matches,
    selected_recipe,
)
from r0.modules.priors import (
    CalibratedPopulationGraph,
    CalibratedPopulationIntervention,
    CalibratedPopulationQuery,
    CausalLimits,
    GPRootCalibration,
    GPRootCalibrationQuery,
    GPRootDAGRecipe,
    TrainingTask,
    explain_generation,
    validate_gp_root_task,
)


@dataclass(frozen=True, slots=True)
class RunSummary:
    attempted: int
    completed: int
    failed: int
    pending: int
    status: str


def _size(root: Path) -> int:
    return sum(p.stat().st_size for p in root.rglob("*") if p.is_file())


def _previous(group: Path, request: RunRequest) -> tuple[set[int], set[int]]:
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
                    if not record_matches(record, request, task_id):
                        raise ValueError("saved group belongs to a different recipe or task")
            if request.observation is not None:
                for observed in load_group(archive):
                    request.observation.validate(observed, request.causal_limits or CausalLimits())
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
        return generate_requested(request, task_id)

    factory = produce or generate
    config = json.loads(json.dumps(asdict(request), sort_keys=True))
    if request.sampling is None:
        config.pop("sampling")
    if request.causal_limits is None:
        config.pop("causal_limits")
    if request.construction is None:
        config.pop("construction")
    if request.observation is None:
        config.pop("observation")
    config["format"] = "generation-run-v1" if request.sampling is None else "generation-run-v2"
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
        completed, failed = _previous(group, request)
        stop = min(start + request.group_size, request.start + request.count)
        ids = set(range(start, stop))
        if not completed.issubset(ids) or not failed.issubset(ids):
            raise ValueError("saved group task IDs do not match requested interval")
        pending = sorted(ids - completed - (set() if retry_failed else failed))
        if pending and status == "COMPLETE":
            remaining = request.max_bytes - used_bytes
            # Reserve bounded attempt metadata before writing any more archive data.
            reserve = 4096 + 4096 * len(pending)
            start_record: dict[str, object] = {"task_ids": pending}
            if request.sampling is not None:
                start_record["recipes"] = {
                    str(task_id): asdict(selected_recipe(request, task_id)) for task_id in pending
                }
                # Include the complete proposal record before admitting any task.
                reserve += len(json.dumps(start_record, allow_nan=False).encode()) + 1
            if remaining < reserve + 4096:
                status = "STORAGE_LIMIT"
            elif time.monotonic() >= deadline:
                status = "TIME_LIMIT"
            else:
                group.mkdir(parents=True, exist_ok=True)
                previous_bytes = _size(group)
                attempt = len(list(group.glob("start-*.json")))
                write_record(group / f"start-{attempt}.json", start_record)
                if request.sampling is not None and persist is not None:
                    persist()
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
                            not record_matches(record, request, task_id)
                            or task.inputs.device.type != request.device
                        ):
                            raise ValueError("producer returned a different task or recipe")
                        source_task = (
                            request.observation.validate(
                                task, request.causal_limits or CausalLimits()
                            )
                            if request.observation is not None
                            else task
                        )
                        if record["recipe"].get("version") == "population-xgboost-graph-v1":
                            from r0.modules.priors.lib.xgboost_population_task import (
                                validate_xgboost_population_task,
                            )

                            validate_xgboost_population_task(
                                source_task, limits=request.causal_limits or CausalLimits()
                            )
                        if record["recipe"].get("version") in (
                            "gp-root-dag-v1",
                            "gp-root-calibration-v1",
                            "gp-root-calibration-query-v1",
                        ):
                            cap = request.causal_limits or (
                                request.recipe.limits
                                if isinstance(
                                    request.recipe,
                                    (GPRootDAGRecipe, GPRootCalibration, GPRootCalibrationQuery),
                                )
                                else CausalLimits()
                            )
                            validate_gp_root_task(source_task, limits=cap)
                        if record["recipe"].get("version") in (
                            "calibrated-population-graph-v1",
                            "calibrated-population-query-v1",
                            "calibrated-population-intervention-v1",
                        ):
                            selected = selected_recipe(request, task_id)
                            if isinstance(
                                selected,
                                (
                                    CalibratedPopulationGraph,
                                    CalibratedPopulationIntervention,
                                    CalibratedPopulationQuery,
                                ),
                            ):
                                explain_generation(
                                    source_task,
                                    [0],
                                    limits=request.causal_limits or selected.limits,
                                )
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
