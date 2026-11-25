"""Worker count and storage persistence do not change the requested tasks."""

import zipfile
from pathlib import Path
from threading import Barrier

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
