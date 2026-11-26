"""The credential-free launch command refuses requests before producing a plan."""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def launch(tmp_path: Path) -> tuple[Path, dict[str, str]]:
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    subprocess.run(
        [
            "git",
            "-c",
            "user.name=Test",
            "-c",
            "user.email=test@example.com",
            "commit",
            "-qm",
            "fixture",
            "--allow-empty",
        ],
        cwd=tmp_path,
        check=True,
    )
    sha = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=tmp_path, text=True).strip()
    config = tmp_path / "configs/generation/request.toml"
    config.parent.mkdir(parents=True)
    config.write_text((ROOT / "configs/generation/smoke-cpu.toml").read_text())
    (tmp_path / "history.json").write_text("[]")
    env = {
        **os.environ,
        "GITHUB_SHA": sha,
        "GITHUB_RUN_ID": "123",
        "GITHUB_RUN_ATTEMPT": "1",
        "MAX_GENERATION_TASKS": "8",
        "MAX_GENERATION_SECONDS": "60",
        "MAX_GENERATION_BYTES": "10485760",
        "MAX_GPU": "none",
        "MAX_TRAIN_RUNS_PER_DAY": "2",
        "GENERATION_RUN_RESERVATION_USD": "2.5",
        "GENERATION_DAILY_BUDGET_USD": "5",
    }
    return tmp_path, env


def invoke(root: Path, env: dict[str, str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/check_generation_caps.py"),
            "--config",
            "configs/generation/request.toml",
            "--run",
            "test-run",
            "--history",
            "history.json",
            "--out",
            "reservation.json",
        ],
        cwd=root,
        env=env,
        text=True,
        capture_output=True,
    )


def test_launch_reserves_before_execution(launch: tuple[Path, dict[str, str]]) -> None:
    root, env = launch
    result = invoke(root, env)
    assert result.returncode == 0, result.stderr
    plan = json.loads((root / "reservation.json").read_text())
    assert plan["reserved_usd"] == 2.5
    assert plan["actual_cost_usd"] == "NOT_MEASURED"
    assert plan["status"] == "RESERVED_NOT_SUBMITTED"


@pytest.mark.parametrize(
    ("variable", "value"),
    [
        ("GITHUB_RUN_ATTEMPT", "2"),
        ("MAX_GENERATION_TASKS", "7"),
        ("MAX_GENERATION_BYTES", "65536"),
        ("MAX_GENERATION_SECONDS", "0"),
        ("GENERATION_DAILY_BUDGET_USD", "4"),
        ("GENERATION_RUN_RESERVATION_USD", "nan"),
    ],
)
def test_launch_rejects_invalid_or_over_budget_requests(
    launch: tuple[Path, dict[str, str]], variable: str, value: str
) -> None:
    root, env = launch
    result = invoke(root, {**env, variable: value})
    assert result.returncode != 0
    assert not (root / "reservation.json").exists()


@pytest.mark.parametrize(
    "controls",
    [
        'task_type = "binary"\nclasses = 2\nseparation = 2.0',
        'task_type = "multiclass"\nclasses = 4\nclass_offsets = [0, 1, 2, 3]',
    ],
)
def test_classification_keeps_its_controls_and_has_valid_agreement_recipes(
    launch: tuple[Path, dict[str, str]], controls: str
) -> None:
    root, env = launch
    config = root / "configs/generation/request.toml"
    config.write_text(config.read_text().replace("noise_ratio = 0.5", controls))
    result = invoke(root, env)
    assert result.returncode == 0, result.stderr
    plan = json.loads((root / "reservation.json").read_text())
    assert plan["request"]["recipe"]["task_type"] != "regression"
    from r0.modules.priors import LinearRecipe

    for case in plan["agreement_cases"]:
        entry = case["recipe"]
        entry["class_offsets"] = tuple(entry["class_offsets"])
        LinearRecipe(**entry)


@pytest.mark.parametrize("family", ["fourier", "periodic", "polynomial"])
def test_admission_diagnostics_preserve_the_requested_family(
    launch: tuple[Path, dict[str, str]], family: str
) -> None:
    from r0.modules.priors import FourierRecipe, PolynomialRecipe, recipe_from_dict

    root, env = launch
    (root / "configs/generation/request.toml").write_text(
        (ROOT / f"configs/generation/{family}-cpu.toml").read_text()
    )
    result = invoke(root, env)
    assert result.returncode == 0, result.stderr
    plan = json.loads((root / "reservation.json").read_text())
    requested = recipe_from_dict(plan["request"]["recipe"])
    assert isinstance(requested, (FourierRecipe, PolynomialRecipe))
    assert requested.task.rows == 128
    assert requested.task.noise_ratio == 0.5
    for case in plan["agreement_cases"]:
        entry = case["recipe"]
        recipe = recipe_from_dict(entry)
        assert type(recipe) is type(requested)
        if isinstance(recipe, FourierRecipe) and isinstance(requested, FourierRecipe):
            assert recipe.family == requested.family
            assert recipe.terms == requested.terms
        if isinstance(recipe, PolynomialRecipe) and isinstance(requested, PolynomialRecipe):
            assert recipe.inputs == requested.inputs
            assert recipe.degree == requested.degree
        assert isinstance(recipe, (FourierRecipe, PolynomialRecipe))
        assert recipe.task.rows == 64
        assert recipe.task.noise_ratio == 0


def test_sampled_request_admission_keeps_family_and_profile(
    launch: tuple[Path, dict[str, str]],
) -> None:
    from r0.modules.generation import RecipeSampling, request_from_dict
    from r0.modules.priors import PolynomialRecipe, recipe_from_dict

    root, env = launch
    config = root / "configs/generation/request.toml"
    config.write_text(
        config.read_text() + '\n[sampling]\nfamily = "polynomial"\nprofile = "broad-v1"\n'
    )
    result = invoke(root, env)
    assert result.returncode == 0, result.stderr
    plan = json.loads((root / "reservation.json").read_text())
    request = request_from_dict(plan["request"])
    assert request.sampling == RecipeSampling("polynomial", "broad-v1")
    task_types: set[str] = set()
    for case in plan["agreement_cases"]:
        entry = case["recipe"]
        recipe = recipe_from_dict(entry)
        assert isinstance(recipe, PolynomialRecipe)
        assert recipe.task.rows == 64
        assert recipe.task.noise_ratio == 0
        task_types.add(recipe.task.task_type)
    assert task_types == {"regression", "binary", "multiclass"}


def test_composed_request_has_executable_agreement_recipes(
    launch: tuple[Path, dict[str, str]],
) -> None:
    from dataclasses import asdict

    from r0.modules.generation import load_request
    from r0.modules.priors import CompositeRecipe, generate_task, recipe_from_dict

    root, env = launch
    config = root / "configs/generation/request.toml"
    config.write_text((ROOT / "configs/generation/composite-cpu.toml").read_text())
    requested = json.loads(json.dumps(asdict(load_request(config))))
    result = invoke(root, env)
    assert result.returncode == 0, result.stderr
    plan = json.loads((root / "reservation.json").read_text())
    assert plan["request"] == requested
    original = recipe_from_dict(requested["recipe"])
    assert isinstance(original, CompositeRecipe)
    task_types: set[str] = set()
    for case in plan["agreement_cases"]:
        entry = case["recipe"]
        recipe = recipe_from_dict(entry)
        assert isinstance(recipe, CompositeRecipe)
        assert recipe.expression == original.expression
        generated = generate_task(recipe, seed=27, task_id=0)
        assert generated.inputs.shape[0] == 64
        task_types.add(generated.task_type)
    assert task_types == {"regression", "binary", "multiclass"}


def test_gpu_plan_budgets_the_full_device_matrix(launch: tuple[Path, dict[str, str]]) -> None:
    root, env = launch
    config = root / "configs/generation/request.toml"
    config.write_text((ROOT / "configs/generation/smoke-gpu.toml").read_text())
    result = invoke(root, {**env, "MAX_GPU": "T4"})
    assert result.returncode == 0, result.stderr
    plan = json.loads((root / "reservation.json").read_text())
    assert plan["agreement_seconds"] == 180
    assert len(plan["agreement_cases"]) == 93
    assert len({case["name"] for case in plan["agreement_cases"]}) == 93
    assert {case["name"] for case in plan["agreement_cases"][:3]} == {
        "requested-regression",
        "requested-binary",
        "requested-multiclass",
    }
    assert plan["estimated_function_usd"] == pytest.approx(840 * 0.00018598)


def test_gpu_agreement_cannot_exceed_workflow_deadline(
    launch: tuple[Path, dict[str, str]],
) -> None:
    root, env = launch
    (root / "configs/generation/request.toml").write_text(
        (ROOT / "configs/generation/smoke-gpu.toml").read_text()
    )
    result = invoke(root, {**env, "MAX_GPU": "T4", "GENERATION_JOB_TIMEOUT_MINUTES": "10"})
    assert result.returncode != 0
    assert "exceed the workflow deadline" in result.stderr
    assert not (root / "reservation.json").exists()
