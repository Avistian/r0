"""Saved comparisons survive a new explorer session without changing the base task."""

import base64
import io
import json
import zipfile
from pathlib import Path

import numpy as np
import pytest

from r0.modules.explorer import ExplorerSession


def test_pair_is_saved_verified_and_reopened_without_regeneration(tmp_path: Path) -> None:
    session = ExplorerSession(tmp_path)
    before = session.request(
        "generate",
        {
            "recipe": {"rows": 32, "context_rows": 20, "noise_ratio": 0.2},
            "seed": str(2**63 - 1),
            "task_id": 9,
        },
    )
    result = session.request("pair", {"intervention": "noise", "value": 0.8})
    pair = result["pair"]
    assert result["view"] == before["view"]
    assert result["archive"] == before["archive"]
    assert pair["after"]["view"]["episode_id"] != before["view"]["episode_id"]
    card = json.loads(pair["card_json"])
    assert card["seed"] == 2**63 - 1
    assert card["after"]["residual_std"] == pytest.approx(4 * card["before"]["residual_std"])
    assert all(card["required_equalities"].values())
    assert pair["card"]["seed"] == str(2**63 - 1)
    attempts = sorted(tmp_path.glob(".attempts/*/*.json"))
    reopened = ExplorerSession(tmp_path).request("open_pair", {"path": pair["path"]})
    assert reopened == result
    assert sorted(tmp_path.glob(".attempts/*/*.json")) == attempts
    replay = session.request("pair", {"intervention": "noise", "value": 0.8})
    assert replay["pair"]["run"]["summary"]["generated"] == 0


def test_pair_export_contains_both_complete_tasks_and_one_combined_limit(tmp_path: Path) -> None:
    session = ExplorerSession(tmp_path)
    session.request(
        "generate",
        {"recipe": {"rows": 128, "context_rows": 80, "features": 16}, "seed": 27, "task_id": 9},
    )
    result = session.request("pair", {"intervention": "irrelevant", "value": 24})
    pair = result["pair"]
    exported = session.request("pair_export", {"expected_pair_id": pair["card"]["pair_id"]})
    payload = base64.b64decode(exported["data"])
    with zipfile.ZipFile(io.BytesIO(payload)) as outer:
        assert set(outer.namelist()) == {"card.json", "before.zip", "after.zip"}
        assert json.loads(outer.read("card.json")) == json.loads(pair["card_json"])
        for side, features in (("before", 16), ("after", 24)):
            with zipfile.ZipFile(io.BytesIO(outer.read(f"{side}.zip"))) as nested:
                with np.load(
                    io.BytesIO(nested.read("predictor.npz")), allow_pickle=False
                ) as arrays:
                    assert arrays["context_x"].shape == (80, features)
                    assert arrays["query_x"].shape == (48, features)
                    assert "query_y" not in arrays
                assert len(nested.read("truth.csv").decode().splitlines()) == 129
                assert len(nested.read("predictor.csv").decode().splitlines()) == 129
    # Each dataset fits this budget, but the combined export does not.
    with zipfile.ZipFile(io.BytesIO(payload)) as outer:
        bound = max(65536, *(len(outer.read(f"{side}.zip")) for side in ("before", "after")))
    assert bound < len(payload)
    with pytest.raises(ValueError, match="output-storage limit"):
        session.request("pair_export", {"max_bytes": bound})
    with pytest.raises(ValueError, match="comparison changed"):
        session.request("pair_export", {"expected_pair_id": "stale"})
    with pytest.raises(ValueError, match="active dataset changed"):
        session.request("pair_export", {"expected_episode_id": "stale"})
    assert session.request("view", {}) == result


def test_failed_pair_or_reopen_preserves_current_dataset_and_comparison(tmp_path: Path) -> None:
    session = ExplorerSession(tmp_path)
    session.request("generate", {"recipe": {"noise_ratio": 0.2}, "seed": 27, "task_id": 9})
    result = session.request("pair", {"intervention": "noise", "value": 0.8})
    with pytest.raises(ValueError, match="original dataset's generation device"):
        session.request("pair", {"intervention": "noise", "value": 0.9, "device": "cuda"})
    with pytest.raises(ValueError, match="Timed out: True"):
        session.request("pair", {"intervention": "noise", "value": 0.9, "max_seconds": 0.001})
    assert session.request("view", {}) == result
    failures = list(tmp_path.glob(".pair-attempts/*-failed.json"))
    assert len(failures) == 1
    failure = json.loads(failures[0].read_text())
    assert failure["status"] == "FAILED"
    assert failure["request"]["recipe"]["noise_ratio"] == 0.2
    assert failure["request"]["pairing"]["value"] == 0.9
    assert failure["before_episode_id"] == result["view"]["episode_id"]
    path = tmp_path / result["pair"]["path"]
    original = path.read_text()
    document = json.loads(original)
    document["card"]["after"]["residual_std"] = 123
    path.write_text(json.dumps(document))
    with pytest.raises(ValueError, match="card disagrees"):
        session.request("open_pair", {"path": str(path)})
    assert session.request("view", {}) == result
    document = json.loads(original)
    document["after_archive"] = "../outside.zip"
    path.write_text(json.dumps(document))
    with pytest.raises(ValueError, match="inside"):
        session.request("open_pair", {"path": str(path)})
    assert session.request("view", {}) == result
    with pytest.raises(ValueError, match="inside"):
        session.request("open_pair", {"path": "../outside.json"})
    path.write_text(original)
    archive = tmp_path / document["before_archive"]
    archive.write_bytes(b"corrupt archive")
    with pytest.raises((ValueError, zipfile.BadZipFile)):
        session.request("open_pair", {"path": str(path)})
    assert session.request("view", {}) == result


def test_sparse_estimate_is_explicit_and_new_dataset_clears_pair(tmp_path: Path) -> None:
    session = ExplorerSession(tmp_path)
    recipe = session.request("examples", {})["examples"]["Polynomial"]
    recipe["task"].update(rows=16, context_rows=8)
    before = session.request("generate", {"recipe": recipe, "seed": 27, "task_id": 9})
    with pytest.raises(ValueError, match="allow_estimated_variance"):
        session.request("pair", {"intervention": "sparsity", "value": 2})
    assert session.request("view", {}) == before
    result = session.request(
        "pair", {"intervention": "sparsity", "value": 2, "allow_estimated_variance": True}
    )
    assert (
        result["pair"]["card"]["variance_control"]
        == "independent_calibration_unit_variance_estimate"
    )
    assert result["pair"]["card"]["after"]["expected_signal_variance"] is None
    assert (
        ExplorerSession(tmp_path).request("open_pair", {"path": result["pair"]["path"]}) == result
    )
    opened = session.request("open", {"archive": before["archive"], "task_id": 9})
    assert opened["pair"] is None
    with pytest.raises(ValueError, match="create or open a comparison"):
        session.request("pair_export", {})
    session.request("open_pair", {"path": result["pair"]["path"]})
    new = session.request("generate", {"recipe": {}, "seed": 27, "task_id": 10})
    assert new["pair"] is None
