"""Outcome accounting must not turn proposals or interrupted writes into output."""

import fcntl
import json
import shutil
from dataclasses import asdict, replace
from pathlib import Path

import pytest

from r0.modules import generation as g
from r0.modules import priors as p


def plan() -> g.CoveragePlan:
    request = g.RunRequest(
        p.LinearRecipe(rows=16, context_rows=8, features=2, signal_features=2, active_features=2),
        seed=77,
        count=1,
        max_seconds=30,
        max_bytes=8000000,
        group_size=2,
        sampling=g.RecipeSampling("polynomial"),
    )
    return g.CoveragePlan((g.CoverageStratum("sampled", request),), total=3, shard_size=3)


def interrupted_fixture(out: Path) -> Path:
    campaign = plan()
    out.mkdir(exist_ok=True)
    (out / "coverage-plan.json").write_text(campaign.to_json())
    root = out / "shards" / "00000000"
    root.mkdir(parents=True)
    (root / ".writer.lock").touch()
    request = g.coverage_shard(campaign, 0).request
    (root / "run.json").write_text(json.dumps({**asdict(request), "format": "generation-run-v2"}))
    group = root / "groups" / "0" / "0"
    group.mkdir(parents=True)
    (group / "start-0.json").write_text(json.dumps({"task_ids": [0, 1]}))
    (group / "failures-0.jsonl").write_text(
        json.dumps({"task_id": 0, "error": "fixture failure"}) + "\n"
    )
    (group / "data-0.zip").write_bytes(b"interrupted archive")
    return root


def test_unstarted_and_interrupted_accounting(tmp_path: Path) -> None:
    before = g.inspect_coverage_shard(plan(), 0, tmp_path)
    assert before["pending"] == 3
    assert before["committed"] == 0
    assert not tmp_path.joinpath("shards").exists()
    interrupted_fixture(tmp_path)
    report = g.inspect_coverage_shard(plan(), 0, tmp_path)
    assert report["requested"] == 3
    assert report["committed"] == 0
    assert report["failed_unresolved"] == 1
    assert report["pending"] == 2
    assert report["admitted_without_outcome"] == 1
    assert report["incomplete_archives"] == 1
    assert report["status"] == "INCOMPLETE"
    assert report["mechanism_diversity"] == "NOT_MEASURED"


def test_verified_output_recovery_and_duplicate_detection(tmp_path: Path) -> None:
    result = g.run_coverage_shard(plan(), 0, tmp_path)
    assert result.returncode == 0, result.stderr
    report = g.inspect_coverage_shard(plan(), 0, tmp_path)
    assert report["committed"] == 3
    assert report["pending"] == 0
    assert report["verified_archive_bytes"] > 0
    assert report["status"] == "COMPLETE"
    archive = next(tmp_path.rglob("data-*.zip"))
    # A prior failure remains recorded even when a later attempt succeeded.
    (archive.parent / "failures-historical.jsonl").write_text(
        json.dumps({"task_id": 0, "error": "recovered fixture failure"}) + "\n"
    )
    recovered = g.inspect_coverage_shard(plan(), 0, tmp_path)
    assert recovered["failed_then_committed"] == 1
    assert recovered["failed_unresolved"] == 0
    duplicate = archive.with_name("data-duplicate.zip")
    shutil.copyfile(archive, duplicate)
    shutil.copyfile(archive.with_suffix(".json"), duplicate.with_suffix(".json"))
    with pytest.raises(ValueError, match="duplicate committed"):
        g.inspect_coverage_shard(plan(), 0, tmp_path)


def test_refuses_wrong_plan_header_ids_and_live_writer(tmp_path: Path) -> None:
    root = interrupted_fixture(tmp_path)
    with pytest.raises(ValueError, match="different plan"):
        g.inspect_coverage_shard(replace(plan(), total=4), 0, tmp_path)
    with (root / ".writer.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(ValueError, match="actively written"):
            g.inspect_coverage_shard(plan(), 0, tmp_path)
    start = root / "groups" / "0" / "0" / "start-0.json"
    start.write_text(json.dumps({"task_ids": [100]}))
    with pytest.raises(ValueError, match="outside"):
        g.inspect_coverage_shard(plan(), 0, tmp_path)
    header = root / "run.json"
    data = json.loads(header.read_text())
    data["seed"] += 1
    header.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="run header"):
        g.inspect_coverage_shard(plan(), 0, tmp_path)


def test_campaign_keeps_unstarted_shards_in_denominator(tmp_path: Path) -> None:
    campaign = replace(plan(), total=5, shard_size=3)
    result = g.run_coverage_shard(campaign, 0, tmp_path)
    assert result.returncode == 0, result.stderr
    report = g.inspect_coverage(campaign, tmp_path)
    assert report["totals"]["requested"] == 5
    assert report["totals"]["committed"] == 3
    assert report["totals"]["pending"] == 2
    assert report["inspected_shards"] == 1
    assert report["status"] == "INCOMPLETE"
    with pytest.raises(ValueError, match="different plan"):
        g.inspect_coverage(plan(), tmp_path)
