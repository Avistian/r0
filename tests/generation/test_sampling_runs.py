"""Profile settings persist and replay through the public bounded run interface."""

import json
from dataclasses import asdict
from pathlib import Path

from r0.modules.generation import RecipeSampling, RunRequest, load_group, run_generation
from r0.modules.priors import LinearRecipe, TrainingTask, generate_task, sample_recipe


def test_sampled_run_saves_selected_recipes_and_resumes_without_generation(tmp_path: Path) -> None:
    controls = LinearRecipe(rows=32, context_rows=20)
    request = RunRequest(
        controls,
        seed=27,
        count=3,
        group_size=2,
        max_seconds=20,
        max_bytes=2_000_000,
        sampling=RecipeSampling("polynomial"),
    )
    first = run_generation(request, tmp_path, workers=2)
    assert first.completed == 3
    assert first.failed == 0
    saved = [task for path in sorted(tmp_path.rglob("data-*.zip")) for task in load_group(path)]
    assert len(saved) == 3
    degrees: set[int] = set()
    for task in saved:
        record = json.loads(task.recipe_json)
        task_id = record["task_id"]
        selected = sample_recipe("polynomial", task=controls, seed=27, task_id=task_id)
        assert record["recipe"] == json.loads(json.dumps(asdict(selected)))
        assert record["sampling"] == {
            "family": "polynomial",
            "profile": "broad-v1",
            "task": json.loads(json.dumps(asdict(controls))),
        }
        degrees.add(record["recipe"]["degree"])
        assert task.episode() == generate_task(selected, seed=27, task_id=task_id).episode()
    assert len(degrees) > 1
    for start in tmp_path.rglob("start-*.json"):
        proposal = json.loads(start.read_text())
        assert set(proposal["recipes"]) == set(map(str, proposal["task_ids"]))

    def unexpected(task_id: int) -> TrainingTask:
        raise AssertionError(f"regenerated task {task_id}")

    second = run_generation(request, tmp_path, produce=unexpected)
    assert second.completed == 3
    assert second.attempted == 0


def test_profile_request_round_trips_through_toml_and_guarded_worker(tmp_path: Path) -> None:
    from r0.modules.generation import load_request, request_from_dict, run_guarded

    config = tmp_path / "sample.toml"
    config.write_text("""
[recipe]
rows = 32
context_rows = 20
task_type = "binary"
classes = 2
[sampling]
family = "polynomial"
profile = "broad-v1"
[run]
seed = 27
count = 2
group_size = 2
max_seconds = 30
max_bytes = 2000000
""")
    request = load_request(config)
    snapshot = json.loads(json.dumps(asdict(request)))
    assert request_from_dict(snapshot) == request
    assert snapshot == json.loads(json.dumps(asdict(request)))
    first = run_guarded(request, tmp_path / "run", workers=2)
    assert first.returncode == 0, first.stderr
    assert first.summary is not None
    assert first.summary["generated"] == 2
    second = run_guarded(request, tmp_path / "run", workers=1)
    assert second.returncode == 0, second.stderr
    assert second.summary is not None
    assert second.summary["generated"] == 0
    assert second.summary["completed"] == 2


def test_failed_sampled_tasks_keep_proposals_and_only_retry_explicitly(tmp_path: Path) -> None:
    controls = LinearRecipe(rows=32, context_rows=20)
    request = RunRequest(
        controls,
        seed=27,
        count=2,
        max_seconds=20,
        max_bytes=2_000_000,
        sampling=RecipeSampling("polynomial"),
    )
    persisted_before_admission: list[int] = []
    called: list[int] = []

    def persist() -> None:
        persisted_before_admission.append(len(called))

    def fail(task_id: int) -> TrainingTask:
        called.append(task_id)
        proposal = json.loads((tmp_path / "groups/0/0/start-0.json").read_text())
        selected = sample_recipe("polynomial", task=controls, seed=27, task_id=task_id)
        assert proposal["recipes"][str(task_id)] == json.loads(json.dumps(asdict(selected)))
        raise ValueError("authored failure after recipe was retained")

    first = run_generation(request, tmp_path, produce=fail, persist=persist)
    assert persisted_before_admission[0] == 0
    assert first.failed == 2
    assert first.completed == 0
    second = run_generation(request, tmp_path)
    assert second.attempted == 0
    assert second.failed == 2
    recovered = run_generation(request, tmp_path, retry_failed=True)
    assert recovered.completed == 2
    assert recovered.attempted == 2
    assert "authored failure" in (tmp_path / "groups/0/0/failures-0.jsonl").read_text()
    assert json.loads((tmp_path / "groups/0/0/start-0.json").read_text()) == json.loads(
        (tmp_path / "groups/0/0/start-1.json").read_text()
    )


def test_sampling_proposal_bytes_are_reserved_before_admission(tmp_path: Path) -> None:
    controls = LinearRecipe(features=256, signal_features=256, active_features=256)
    request = RunRequest(
        controls,
        seed=27,
        count=8,
        group_size=8,
        max_seconds=20,
        max_bytes=65536,
        sampling=RecipeSampling("composition"),
    )

    def unexpected(task_id: int) -> TrainingTask:
        raise AssertionError(f"task {task_id} was admitted without proposal space")

    result = run_generation(request, tmp_path, produce=unexpected)
    assert result.status == "STORAGE_LIMIT"
    assert result.attempted == 0
    assert sum(p.stat().st_size for p in tmp_path.rglob("*") if p.is_file()) <= request.max_bytes


def test_matching_profile_name_cannot_disguise_a_wrong_selected_recipe(tmp_path: Path) -> None:
    from dataclasses import replace

    controls = LinearRecipe(rows=32, context_rows=20)
    request = RunRequest(
        controls,
        seed=27,
        count=1,
        max_seconds=20,
        max_bytes=2_000_000,
        sampling=RecipeSampling("polynomial"),
    )

    def wrong(task_id: int) -> TrainingTask:
        task = generate_task(controls, seed=27, task_id=task_id)
        record = json.loads(task.recipe_json)
        record["sampling"] = {
            "family": "polynomial",
            "profile": "broad-v1",
            "task": asdict(controls),
        }
        return replace(task, recipe_json=json.dumps(record))

    result = run_generation(request, tmp_path, produce=wrong)
    assert result.failed == 1
    assert result.completed == 0
    assert "different task or recipe" in (tmp_path / "groups/0/0/failures-0.jsonl").read_text()
