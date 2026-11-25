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

    for entry in plan["agreement_recipes"]:
        entry["class_offsets"] = tuple(entry["class_offsets"])
        LinearRecipe(**entry)


@pytest.mark.parametrize("family", ["fourier", "periodic"])
def test_admission_diagnostics_preserve_the_requested_family(
    launch: tuple[Path, dict[str, str]], family: str
) -> None:
    from r0.modules.priors import FourierRecipe, recipe_from_dict

    root, env = launch
    (root / "configs/generation/request.toml").write_text(
        (ROOT / f"configs/generation/{family}-cpu.toml").read_text()
    )
    result = invoke(root, env)
    assert result.returncode == 0, result.stderr
    plan = json.loads((root / "reservation.json").read_text())
    requested = recipe_from_dict(plan["request"]["recipe"])
    assert isinstance(requested, FourierRecipe)
    assert requested.task.rows == 128
    assert requested.task.noise_ratio == 0.5
    for entry in plan["agreement_recipes"]:
        recipe = recipe_from_dict(entry)
        assert isinstance(recipe, FourierRecipe)
        assert recipe.family == requested.family
        assert recipe.terms == requested.terms
        assert recipe.task.rows == 64
        assert recipe.task.noise_ratio == 0
