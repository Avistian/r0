"""The UI generates through the bounded runner and reopens its saved output."""

import base64
import io
import json
import zipfile
from pathlib import Path

import pytest

from r0.modules.explorer import ExplorerSession
from r0.modules.priors import LinearRecipe, generate_task


def test_session_generates_reopens_inspects_and_exports(tmp_path: Path) -> None:
    session = ExplorerSession(tmp_path)
    result = session.request(
        "generate",
        {
            "recipe": {"rows": 32, "context_rows": 20},
            "seed": 27,
            "task_id": 9,
            "max_seconds": 30,
            "max_bytes": 1048576,
        },
    )
    assert result["view"]["total_rows"] == 32
    assert result["run"]["summary"]["completed"] == 1
    identity = result["view"]["source_task_id"]
    replay = session.request(
        "generate", {"record": result["view"]["recipe"], "max_seconds": 30, "max_bytes": 1048576}
    )
    assert replay["view"]["source_task_id"] == identity
    assert replay["run"]["summary"]["generated"] == 0
    reopened = ExplorerSession(tmp_path).request(
        "open", {"archive": result["archive"], "task_id": 9}
    )
    assert reopened["view"] == result["view"]
    assert session.request("row", {"index": 31})["row"]["index"] == 31
    curve = session.request("curve", {"feature": 0, "anchor_row": 31, "points": 5})["curve"]
    assert curve["episode_id"] == result["view"]["episode_id"]
    assert len(curve["scores"]) == 5
    exported = session.request("export", {})
    with zipfile.ZipFile(io.BytesIO(base64.b64decode(exported["data"]))) as archive:
        assert len(archive.read("truth.csv").decode().splitlines()) == 33
    with pytest.raises(ValueError, match="invalid sizes"):
        session.request("generate", {"recipe": {"rows": 0}, "seed": 27, "task_id": 9})
    assert session.request("view", {})["view"]["source_task_id"] == identity
    with pytest.raises(ValueError, match="inside"):
        session.request("open", {"archive": "../outside.zip", "task_id": 9})
    session.request(
        "generate", {"recipe": {"rows": 32, "context_rows": 20}, "seed": 27, "task_id": 10}
    )
    with pytest.raises(ValueError, match="active dataset changed"):
        session.request("export", {"expected_episode_id": result["view"]["episode_id"]})


def test_browser_identities_keep_all_63_bits(tmp_path: Path) -> None:
    seed, task_id, split = 2**63 - 1, 2**53 + 123, 2**53 + 125
    record = {"recipe": {"split": split}, "seed": seed, "task_id": task_id}
    session = ExplorerSession(tmp_path)
    result = session.request("generate", {"record_json": json.dumps(record), "max_seconds": 30})
    expected = generate_task(LinearRecipe(split=split), seed=seed, task_id=task_id)
    assert result["view"]["source_task_id"] == expected.source_task_id
    assert result["view"]["episode_id"] == expected.episode_id
    assert result["view"]["recipe"]["seed"] == str(seed)
    assert result["view"]["recipe"]["recipe"]["split"] == str(split)
    assert json.loads(result["recipe_json"])["task_id"] == task_id
    replay = session.request("generate", {"record": result["view"]["recipe"], "max_seconds": 30})
    assert replay["view"]["episode_id"] == expected.episode_id
    assert replay["run"]["summary"]["generated"] == 0


def test_startup_timeout_retains_request_and_failure_before_worker_initialization(
    tmp_path: Path,
) -> None:
    session = ExplorerSession(tmp_path)
    with pytest.raises(ValueError, match="Timed out: True"):
        session.request("generate", {"recipe": {}, "seed": 27, "task_id": 12, "max_seconds": 0.001})
    attempts = list(tmp_path.glob(".attempts/*/*.json"))
    assert len(attempts) == 2
    records = [json.loads(path.read_text()) for path in attempts]
    assert {record["status"] for record in records} == {"STARTED", "FAILED"}
    failure = next(record for record in records if record["status"] == "FAILED")
    assert failure["request"]["start"] == 12
    assert failure["request"]["seed"] == 27
    assert failure["request"]["recipe"]["version"] == "linear-v1"
    assert failure["result"]["timed_out"] is True
