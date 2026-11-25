"""Shared bounded generation and verified local archives for explorer actions."""

import json
import os
import uuid
from dataclasses import asdict
from pathlib import Path
from typing import Any

from r0.modules.generation import RunRequest, load_group, run_guarded
from r0.modules.priors import TrainingTask, record_fingerprint


def retain_record(path: Path, record: dict[str, Any]) -> None:
    pending = path.with_suffix(".pending")
    with pending.open("x") as stream:
        json.dump(record, stream, allow_nan=False, sort_keys=True)
        stream.flush()
        os.fsync(stream.fileno())
    pending.rename(path)


def inside(output_dir: Path, path: Path) -> Path:
    path = path.resolve()
    if not path.is_relative_to(output_dir):
        raise ValueError("saved path must be inside the explorer output directory")
    return path


def load_saved(output_dir: Path, path: Path, task_id: int) -> TrainingTask:
    for task in load_group(inside(output_dir, path)):
        if json.loads(task.recipe_json)["task_id"] == task_id:
            return task
    raise ValueError("task identifier was not found in this archive")


def run_saved(output_dir: Path, request: RunRequest) -> tuple[TrainingTask, str, dict[str, Any]]:
    config = asdict(request)
    if request.sampling is None:
        config.pop("sampling")
    out = output_dir / record_fingerprint(config)
    attempts = output_dir / ".attempts" / out.name
    attempts.mkdir(parents=True, exist_ok=True)
    attempt = uuid.uuid4().hex
    record = {"attempt": attempt, "request": asdict(request), "output": str(out)}
    retain_record(attempts / f"{attempt}-started.json", {**record, "status": "STARTED"})
    try:
        run = run_guarded(request, out)
    except (OSError, RuntimeError, ValueError, KeyboardInterrupt) as error:
        retain_record(
            attempts / f"{attempt}-failed.json",
            {**record, "status": "FAILED", "error": repr(error)},
        )
        raise
    failed = run.timed_out or bool(run.returncode) or run.summary is None
    retain_record(
        attempts / f"{attempt}-result.json",
        {**record, "status": "FAILED" if failed else "COMPLETE", "result": asdict(run)},
    )
    if failed:
        raise ValueError(
            f"Generation failed; attempt records retained in {attempts}. "
            f"Timed out: {run.timed_out}. {run.stderr[-2000:]} {run.summary}"
        )
    archives = sorted(out.glob("groups/**/*.zip"))
    if len(archives) != 1:
        raise ValueError(f"Expected one completed task archive in {out}")
    task = load_saved(output_dir, archives[0], request.start)
    return task, str(archives[0].relative_to(output_dir)), asdict(run)
