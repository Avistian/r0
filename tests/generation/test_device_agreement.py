"""Exercise the portable comparison command without claiming CUDA coverage on CPU."""

import hashlib
import json
import os
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path

from r0.modules.priors import LinearRecipe, recipe_from_dict

ROOT = Path(__file__).resolve().parents[2]


def test_cpu_comparison_records_the_actual_devices_and_case_result(tmp_path: Path) -> None:
    source = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    plan = tmp_path / "plan.json"
    plan.write_text(
        json.dumps(
            {
                "source": source,
                "request": {"seed": 27, "start": 5},
                "agreement_cases": [
                    {
                        "name": "small-linear-regression",
                        "recipe": asdict(LinearRecipe(rows=16, context_rows=8, noise_ratio=0.5)),
                    }
                ],
            }
        )
    )
    out = tmp_path / "comparison"
    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/check_device_agreement.py"),
            "--plan",
            str(plan),
            "--out",
            str(out),
            "--candidate",
            "cpu",
        ],
        env={
            **os.environ,
            "OMP_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
        },
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    manifest = json.loads((out / "manifest.json").read_text())
    assert manifest["plan_sha256"] == hashlib.sha256(plan.read_bytes()).hexdigest()
    assert hashlib.sha256((out / "plan.json").read_bytes()).hexdigest() == manifest["plan_sha256"]
    assert manifest["reference_device"] == manifest["candidate_device"] == "cpu"
    case = json.loads((out / "case-0.json").read_text())
    assert case["status"] == "PASS"
    assert case["name"] == "small-linear-regression"
    assert set(case["max_absolute_error"].values()) == {0.0}
    assert case["source_task_id"]
    summary = json.loads((out / "summary.json").read_text())
    assert summary["status"] == "PASS"
    assert summary["comparison"] == "cpu_to_cpu"
    assert summary["passed"] == summary["planned"] == 1
    assert summary["failed"] == summary["not_run"] == 0


def test_matrix_plan_covers_each_family_size_and_target_without_generation(tmp_path: Path) -> None:
    out = tmp_path / "matrix"
    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/check_device_agreement.py"),
            "--out",
            str(out),
            "--candidate",
            "cuda",
            "--plan-only",
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    plan = json.loads((out / "plan.json").read_text())
    families = {
        "linear",
        "grouped",
        "polynomial",
        "quadratic",
        "rff",
        "periodic",
        "radial",
        "sinusoidal",
        "gp",
        "composition",
    }
    expected = {
        f"matrix-{family}-{size}-{task_type}"
        for family in families
        for size in ("small", "medium", "large")
        for task_type in ("regression", "binary", "multiclass")
    }
    assert {case["name"] for case in plan["agreement_cases"]} == expected
    assert len(plan["agreement_cases"]) == 90
    for case in plan["agreement_cases"]:
        _, family, size, kind = case["name"].split("-")
        recipe = recipe_from_dict(case["recipe"])
        task = recipe if isinstance(recipe, LinearRecipe) else recipe.task
        assert task.features == {"small": 8, "medium": 32, "large": 256}[size]
        expected_rows = {"small": 64, "medium": 1024, "large": 20000}[size]
        if family == "gp":
            expected_rows = {"small": 64, "medium": 512, "large": 1024}[size]
        assert task.rows == expected_rows
        assert task.task_type == kind
        if kind == "multiclass":
            assert task.classes == (100 if size == "large" else 3)
    summary = json.loads((out / "summary.json").read_text())
    assert summary["status"] == "PLAN_ONLY"
    assert summary["not_run"] == 90
    assert summary["passed"] == summary["failed"] == 0
    assert not list(out.glob("case-*.json"))


def test_failed_case_stops_comparison_and_preserves_remaining_count(tmp_path: Path) -> None:
    source = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    valid = asdict(LinearRecipe(rows=16, context_rows=8))
    invalid = {**valid, "rows": 0}
    plan = tmp_path / "plan.json"
    plan.write_text(
        json.dumps(
            {
                "source": source,
                "request": {"seed": 27, "start": 0},
                "agreement_cases": [
                    {"name": "valid", "recipe": valid},
                    {"name": "invalid", "recipe": invalid},
                    {"name": "unattempted", "recipe": valid},
                ],
            }
        )
    )
    out = tmp_path / "failed"
    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/check_device_agreement.py"),
            "--plan",
            str(plan),
            "--out",
            str(out),
            "--candidate",
            "cpu",
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 1, result.stderr
    assert json.loads((out / "case-0.json").read_text())["status"] == "PASS"
    failed = json.loads((out / "case-1.json").read_text())
    assert failed["status"] == "FAILED"
    assert failed["error"]
    assert not (out / "case-2-started.json").exists()
    summary = json.loads((out / "summary.json").read_text())
    assert summary["status"] == "FAILED"
    assert summary["passed"] == summary["failed"] == summary["not_run"] == 1


def test_unavailable_cuda_never_falls_back_to_cpu(tmp_path: Path) -> None:
    out = tmp_path / "cuda"
    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/check_device_agreement.py"),
            "--out",
            str(out),
            "--candidate",
            "cuda",
        ],
        env={**os.environ, "CUDA_VISIBLE_DEVICES": ""},
        capture_output=True,
        text=True,
    )
    assert result.returncode == 1, result.stderr
    manifest = json.loads((out / "manifest.json").read_text())
    assert manifest["candidate_device"] == "cuda"
    assert manifest["gpu"] is None
    summary = json.loads((out / "summary.json").read_text())
    assert summary["status"] == "CUDA_UNAVAILABLE"
    assert summary["planned"] == summary["not_run"] == 90
    assert summary["passed"] == summary["failed"] == 0
    assert not list(out.glob("case-*.json"))
