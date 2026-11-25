"""The notebook/job interface enforces the deadline outside generator threads."""

from pathlib import Path

from r0.modules.generation import RunRequest, run_guarded
from r0.modules.priors import LinearRecipe


def test_guarded_job_completes_and_reports_saved_tasks(tmp_path: Path) -> None:
    request = RunRequest(LinearRecipe(), seed=3, count=2, max_seconds=20, max_bytes=2_000_000)
    result = run_guarded(request, tmp_path, workers=2)
    assert result.returncode == 0, result.stderr
    assert result.summary is not None
    assert result.summary["completed"] == 2
    assert not result.timed_out


def test_guarded_job_kills_the_child_when_startup_exceeds_deadline(tmp_path: Path) -> None:
    request = RunRequest(LinearRecipe(), seed=3, count=2, max_seconds=0.001, max_bytes=2_000_000)
    result = run_guarded(request, tmp_path)
    assert result.timed_out
    assert result.returncode != 0
    assert result.summary is None
