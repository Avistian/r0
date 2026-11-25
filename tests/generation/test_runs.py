"""Generation run recovery exercises real saved outputs."""

import shutil
from pathlib import Path

import pytest

from r0.modules.generation import RunRequest, run_generation
from r0.modules.priors import LinearRecipe, TrainingTask, generate_linear


def test_resume_skips_verified_tasks_and_retries_only_when_requested(tmp_path: Path) -> None:
    recipe = LinearRecipe()
    request = RunRequest(recipe, seed=4, count=3, group_size=2, max_seconds=20, max_bytes=2_000_000)
    called: list[int] = []

    def produce(task_id: int) -> TrainingTask:
        called.append(task_id)
        if task_id == 1:
            raise ValueError("authored numerical failure")
        return generate_linear(recipe, seed=4, task_id=task_id)

    first = run_generation(request, tmp_path, produce=produce)
    assert first.completed == 2
    assert first.failed == 1
    assert called == [0, 1, 2]
    called.clear()
    second = run_generation(request, tmp_path, produce=produce)
    assert called == []
    assert second.completed == 2
    assert second.failed == 1
    third = run_generation(request, tmp_path, retry_failed=True)
    assert third.attempted == 1
    assert third.completed == 3
    assert third.failed == 0
    assert "authored numerical failure" in (tmp_path / "groups/0/0/attempt-0.json").read_text()


def test_restart_recovers_an_interrupted_group_without_overwriting_attempt(tmp_path: Path) -> None:
    recipe = LinearRecipe()
    request = RunRequest(recipe, seed=4, count=2, group_size=2, max_seconds=20, max_bytes=2_000_000)

    def interrupted(task_id: int) -> TrainingTask:
        if task_id == 1:
            raise KeyboardInterrupt("simulated process interruption")
        return generate_linear(recipe, seed=4, task_id=task_id)

    with pytest.raises(KeyboardInterrupt):
        run_generation(request, tmp_path, produce=interrupted)
    archive = tmp_path / "groups/0/0/data-0.zip"
    original = archive.read_bytes()
    result = run_generation(request, tmp_path)
    assert result.completed == 2
    assert result.attempted == 2
    assert archive.read_bytes() == original
    assert (tmp_path / "groups/0/0/start-0.json").exists()
    assert (tmp_path / "groups/0/0/attempt-1.json").exists()


def test_resume_refuses_corrupted_completed_output(tmp_path: Path) -> None:
    request = RunRequest(LinearRecipe(), seed=4, count=2, max_seconds=20, max_bytes=2_000_000)
    run_generation(request, tmp_path)
    archive = tmp_path / "groups/0/0/data-0.zip"
    archive.write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="checksum"):
        run_generation(request, tmp_path)
    assert not (tmp_path / "groups/0/0/start-1.json").exists()


def test_storage_limit_keeps_attempt_and_does_not_claim_completion(tmp_path: Path) -> None:
    recipe = LinearRecipe(rows=1024, context_rows=512)
    request = RunRequest(recipe, seed=4, count=2, max_seconds=20, max_bytes=65_536)
    result = run_generation(request, tmp_path)
    assert result.status == "STORAGE_LIMIT"
    assert result.completed == 0
    assert result.pending == 2
    assert sum(p.stat().st_size for p in tmp_path.rglob("*") if p.is_file()) <= request.max_bytes


def test_zero_or_nonfinite_caps_are_rejected() -> None:
    with pytest.raises(ValueError, match="positive"):
        RunRequest(LinearRecipe(), seed=4, count=2, max_seconds=0, max_bytes=65_536)
    with pytest.raises(ValueError, match="positive"):
        RunRequest(LinearRecipe(), seed=4, count=2, max_seconds=float("nan"), max_bytes=65_536)


def test_valid_archive_from_another_recipe_is_not_accepted(tmp_path: Path) -> None:
    recipe = LinearRecipe()
    first = RunRequest(recipe, seed=4, count=1, max_seconds=20, max_bytes=2_000_000)
    second = RunRequest(recipe, seed=9, count=1, max_seconds=20, max_bytes=2_000_000)
    run_generation(first, tmp_path / "first")
    run_generation(second, tmp_path / "second")
    for suffix in ("zip", "json"):
        shutil.copyfile(
            tmp_path / f"second/groups/0/0/data-0.{suffix}",
            tmp_path / f"first/groups/0/0/data-0.{suffix}",
        )
    with pytest.raises(ValueError, match="different recipe"):
        run_generation(first, tmp_path / "first")


def test_failure_is_retained_when_a_later_task_interrupts_the_group(tmp_path: Path) -> None:
    recipe = LinearRecipe()
    request = RunRequest(recipe, seed=4, count=2, max_seconds=20, max_bytes=2_000_000)

    def interrupted(task_id: int) -> TrainingTask:
        if task_id == 0:
            raise ValueError("failure before interruption")
        raise KeyboardInterrupt("interrupted")

    with pytest.raises(KeyboardInterrupt):
        run_generation(request, tmp_path, produce=interrupted)
    called: list[int] = []

    def recovered(task_id: int) -> TrainingTask:
        called.append(task_id)
        return generate_linear(recipe, seed=4, task_id=task_id)

    result = run_generation(request, tmp_path, produce=recovered)
    assert called == [1]
    assert result.failed == 1
    assert any("failure before interruption" in p.read_text() for p in tmp_path.rglob("*.jsonl"))


@pytest.mark.parametrize("message", ["\u2603" * 1024, "\x00" * 1024])
def test_escaped_error_records_respect_total_storage_limit(tmp_path: Path, message: str) -> None:
    request = RunRequest(LinearRecipe(), seed=4, count=11, max_seconds=20, max_bytes=65_536)

    def fail(_: int) -> TrainingTask:
        raise ValueError(message)

    result = run_generation(request, tmp_path, produce=fail)
    assert result.completed == 0
    assert sum(p.stat().st_size for p in tmp_path.rglob("*") if p.is_file()) <= request.max_bytes
