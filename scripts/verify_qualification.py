"""Audit retained measurements and saved arrays without generating any tasks."""

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

import torch

from r0.modules.generation import load_group


def main(root):
    manifest = json.loads((root / "manifest.json").read_text())
    report = json.loads((root / "report.json").read_text())
    assert report["source"] == manifest["source"]
    for name, expected in manifest["drivers"].items():
        data = subprocess.check_output(["git", "show", f"{manifest['source']}:scripts/{name}"])
        assert hashlib.sha256(data).hexdigest() == expected
    assert len({case["id"] for case in manifest["cases"]}) == len(manifest["cases"])
    identities, groups = set(), {}
    for case in manifest["cases"]:
        directory = root / case["id"]
        requested = json.loads((directory / "request.json").read_text())
        assert requested == case["request"]
        result = json.loads((directory / "result.json").read_text())
        assert result["status"] == "PASS", (case["id"], result["status"])
        assert result["returncode"] == 0
        assert not result["timed_out"]
        assert result["receipt_errors"] == {}
        metric = result["worker"]
        assert metric["verification"] == "PASS"
        assert metric["summary"] == {
            "attempted": 1,
            "completed": 1,
            "failed": 0,
            "pending": 0,
            "status": "COMPLETE",
        }
        assert 0 < metric["generation_and_save_seconds"] < result["process_seconds"]
        assert (
            0
            < metric["generation_peak_rss_bytes"]
            <= metric["process_peak_rss_after_verification_bytes"]
        )
        assert metric["address_space_bytes"] == manifest["address_space_bytes"]
        assert metric["threads"] == 1
        archives = list((directory / "run").rglob("data-*.zip"))
        assert len(archives) == 1
        assert archives[0].stat().st_size == metric["archive_bytes"]
        tasks = list(load_group(archives[0]))
        assert len(tasks) == 1
        task = tasks[0]
        record = json.loads(task.recipe_json)
        assert record["recipe"] == requested["recipe"]
        assert record["seed"] == requested["seed"]
        assert record["task_id"] == requested["start"]
        assert task.episode_id == metric["episode_id"]
        assert task.source_task_id not in identities
        identities.add(task.source_task_id)
        controls = requested["recipe"].get("task", requested["recipe"])
        assert tuple(task.inputs.shape) == (controls["rows"], controls["features"])
        assert len(task.context_indices) == controls["context_rows"]
        assert len(json.loads(task.mechanism_json)["active"]) == controls["active_features"]
        assert torch.isfinite(task.inputs).all()
        assert torch.isfinite(task.scores).all()
        assert torch.isfinite(task.targets).all()
        if task.probabilities is not None:
            assert tuple(task.probabilities.shape) == (controls["rows"], controls["classes"])
            assert torch.isfinite(task.probabilities).all()
        key = (case["family"], case["size"], case["task_type"])
        groups.setdefault(key, []).append((case["id"], metric))
    assert report["totals"]["PASS"] == len(identities)
    assert sum(report["totals"].values()) == len(identities)
    assert len(report["cases"]) == len(groups)
    for row in report["cases"]:
        metrics = groups[(row["family"], row["size"], row["task_type"])]
        assert len(metrics) == 3
        assert row["planned"] == row["confirmed_launched"] == row["passed"] == 3
        assert row["launch_unknown"] == 0
        assert row["case_ids"] == [name for name, _ in metrics]
        seconds = sorted(metric["generation_and_save_seconds"] for _, metric in metrics)
        assert row["seconds_min_median_max"] == seconds
        assert row["extrapolated_100m_generation_days"] == seconds[1] * 100_000_000 / 86400
        sizes = sorted(metric["archive_bytes"] for _, metric in metrics)
        assert row["extrapolated_100m_archive_bytes"] == sizes[1] * 100_000_000
        assert row["generation_peak_rss_bytes"] == max(
            metric["generation_peak_rss_bytes"] for _, metric in metrics
        )
        assert row["qualification"] == "MEASURED_LOCAL_CPU"
    audit = {
        "status": "PASS",
        "source": manifest["source"],
        "verified_tasks": len(identities),
        "verified_cases": len(groups),
        "new_generations": 0,
        "driver_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "report_sha256": hashlib.sha256((root / "report.json").read_bytes()).hexdigest(),
    }
    with (root / "independent-audit.json").open("x") as stream:
        json.dump(audit, stream, indent=2)
        stream.write("\n")
    print(json.dumps(audit))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    main(parser.parse_args().root)
