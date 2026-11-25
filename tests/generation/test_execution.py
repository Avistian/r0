"""Worker count and storage persistence do not change the requested tasks."""

import time
import zipfile
from pathlib import Path
from threading import Barrier

import pytest

from r0.modules.generation import RunRequest, run_generation
from r0.modules.priors import LinearRecipe, TrainingTask, generate_linear


def test_two_workers_run_concurrently_and_persist_each_group(tmp_path: Path) -> None:
    recipe = LinearRecipe()
    request = RunRequest(recipe, seed=3, count=2, max_seconds=20, max_bytes=2_000_000)
    barrier = Barrier(2)
    snapshots: list[list[str]] = []

    def produce(task_id: int) -> TrainingTask:
        barrier.wait(timeout=5)
        return generate_linear(recipe, seed=3, task_id=task_id)

    def persist() -> None:
        snapshots.append([p.name for p in tmp_path.rglob("data-*.json")])

    result = run_generation(request, tmp_path, workers=2, produce=produce, persist=persist)
    assert result.completed == 2
    assert snapshots[-1] == ["data-0.json"]


def test_worker_count_does_not_change_saved_tasks(tmp_path: Path) -> None:
    request = RunRequest(LinearRecipe(), seed=3, count=4, max_seconds=20, max_bytes=2_000_000)
    run_generation(request, tmp_path / "serial", workers=1)
    run_generation(request, tmp_path / "parallel", workers=4)
    with (
        zipfile.ZipFile(tmp_path / "serial/groups/0/0/data-0.zip") as first,
        zipfile.ZipFile(tmp_path / "parallel/groups/0/0/data-0.zip") as second,
    ):
        assert first.namelist() == second.namelist()
        for name in first.namelist():
            assert first.read(name) == second.read(name)


def test_storage_stop_records_already_admitted_worker_failure(tmp_path: Path) -> None:
    recipe = LinearRecipe(rows=1024, context_rows=512)
    request = RunRequest(recipe, seed=3, count=2, max_seconds=20, max_bytes=65_536)
    barrier = Barrier(2)

    def produce(task_id: int) -> TrainingTask:
        barrier.wait(timeout=5)
        if task_id == 1:
            raise ValueError("parallel failure")
        return generate_linear(recipe, seed=3, task_id=task_id)

    result = run_generation(request, tmp_path, workers=2, produce=produce)
    assert result.attempted == 2
    assert result.failed == 1
    assert result.status == "STORAGE_LIMIT"
    assert any("parallel failure" in p.read_text() for p in tmp_path.rglob("*.jsonl"))


def test_provider_error_stops_admission_and_propagates(tmp_path: Path) -> None:
    request = RunRequest(
        LinearRecipe(), seed=3, count=2, group_size=1, max_seconds=20, max_bytes=2_000_000
    )
    called: list[int] = []
    commits = 0

    def produce(task_id: int) -> TrainingTask:
        called.append(task_id)
        raise ValueError("task failed")

    def persist() -> None:
        nonlocal commits
        commits += 1
        if commits == 1:
            raise ValueError("provider unavailable")

    with pytest.raises(ValueError, match="provider unavailable"):
        run_generation(request, tmp_path, produce=produce, persist=persist)
    assert called == [0]


def test_deadline_reason_survives_a_task_failure(tmp_path: Path) -> None:
    request = RunRequest(LinearRecipe(), seed=3, count=2, max_seconds=0.2, max_bytes=2_000_000)

    def slow_failure(_: int) -> TrainingTask:
        time.sleep(0.25)
        raise ValueError("late failure")

    result = run_generation(request, tmp_path, produce=slow_failure)
    assert result.failed == 1
    assert result.pending == 1
    assert result.status == "TIME_LIMIT"
