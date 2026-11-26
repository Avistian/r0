"""Estimate a larger request from hosted measurements without starting generation."""

import json
import os
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path

import pytest

from r0.modules.generation import load_request

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def request_files(tmp_path: Path) -> tuple[Path, Path]:
    config = tmp_path / "larger.toml"
    baseline_config = ROOT / "configs/generation/sampled-gpu.toml"
    config.write_text(baseline_config.read_text().replace("count = 8", "count = 1000"))
    result = tmp_path / "result.json"
    result.write_text(
        json.dumps(
            {
                "status": "COMPLETE",
                "plan": {
                    "source": "a" * 40,
                    "workflow_run_id": "123",
                    "workers": 2,
                    "request": asdict(load_request(baseline_config)),
                },
                "generation": {
                    "source": "a" * 40,
                    "returncode": 0,
                    "timed_out": False,
                    "summary": {
                        "status": "COMPLETE",
                        "generated": 8,
                        "completed": 8,
                        "attempted": 8,
                        "failed": 0,
                        "pending": 0,
                        "generation_seconds": 4.0,
                        "generated_per_second": 2.0,
                        "hardware": "Tesla T4",
                        "device": "cuda",
                        "torch": "test-version",
                        "peak_rss_bytes": 1000000,
                        "gpu_peak_allocated_bytes": 2000000,
                    },
                },
            }
        )
    )
    return result, config


def invoke(result: Path, config: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/estimate_modal_generation.py"),
            "--result",
            str(result),
            "--config",
            str(config),
        ],
        env={**os.environ, "CUDA_VISIBLE_DEVICES": ""},
        capture_output=True,
        text=True,
    )


def test_estimate_reports_measured_scope_and_larger_request_limits(
    request_files: tuple[Path, Path], tmp_path: Path
) -> None:
    result, config = request_files
    before = {p.name: p.read_bytes() for p in tmp_path.iterdir()}
    command = invoke(result, config)
    assert command.returncode == 0, command.stderr
    report = json.loads(command.stdout)
    assert report["measured_tasks"] == 8
    assert report["measured_tasks_per_second"] == 2.0
    assert report["estimated_generation_seconds"] == 500.0
    assert report["requested_tasks"] == 1000
    assert report["generation_time_limit_seconds"] == 60
    assert report["fits_generation_time_limit"] is False
    assert report["output_limit_bytes"] == 10485760
    assert report["baseline_source"] == "a" * 40
    assert report["baseline_hardware"] == "Tesla T4"
    assert report["workers"] == 2
    assert report["estimated_output_bytes"] == "NOT_MEASURED"
    assert {p.name: p.read_bytes() for p in tmp_path.iterdir()} == before


@pytest.mark.parametrize("mismatch", ["device", "source"])
def test_estimate_rejects_receipts_from_a_different_execution(
    request_files: tuple[Path, Path], mismatch: str
) -> None:
    result, config = request_files
    receipt = json.loads(result.read_text())
    if mismatch == "device":
        receipt["generation"]["summary"]["device"] = "cpu"
    else:
        receipt["generation"]["source"] = "b" * 40
    result.write_text(json.dumps(receipt))
    command = invoke(result, config)
    assert command.returncode != 0
    assert not command.stdout


@pytest.mark.parametrize("invalid", ["resume", "partial", "rate", "recipe"])
def test_estimate_refuses_inapplicable_measurements(
    request_files: tuple[Path, Path], invalid: str
) -> None:
    result, config = request_files
    receipt = json.loads(result.read_text())
    if invalid == "resume":
        receipt["generation"]["summary"].update(generated=0, attempted=0)
    elif invalid == "partial":
        receipt["status"] = "INCOMPLETE"
        receipt["generation"]["summary"].update(failed=1, generated=7, completed=7)
    elif invalid == "rate":
        receipt["generation"]["summary"]["generated_per_second"] = 2000
    else:
        config.write_text(config.read_text().replace("features = 8", "features = 16"))
    result.write_text(json.dumps(receipt))
    command = invoke(result, config)
    assert command.returncode != 0
    assert not command.stdout
