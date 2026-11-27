"""Bounded causal workflow preserves active datasets and completed comparisons on failure."""

import base64
import io
import json
import zipfile
from dataclasses import asdict
from pathlib import Path
from typing import Any

import pytest

from r0.modules import priors as p
from r0.modules.explorer import ExplorerSession
from r0.modules.generation import load_causal_pair


def original(session: ExplorerSession) -> dict[str, Any]:
    base = p.FunctionalCausalRecipe(
        p.LinearRecipe(rows=16, context_rows=8, features=2, signal_features=2, active_features=2),
        (
            p.FunctionalCausalNode(
                (), p.FrozenLinearFunction(((1.0,),)), p.JointRootBase("gaussian", 1)
            ),
            p.FunctionalCausalNode((0,), p.FrozenLinearFunction(((2.0,),)), noise_scale=(0.2,)),
            p.FunctionalCausalNode((0, 1), p.FrozenLinearFunction(((-1.0, 3.0),))),
        ),
        ((0, 0), (1, 0)),
        ((2, 0),),
    )
    return session.request(
        "generate",
        {
            "recipe": asdict(base),
            "seed": "9007199254740997",
            "task_id": "9007199254740999",
            "max_seconds": 30,
            "max_bytes": 2_000_000,
        },
    )


def change() -> dict[str, Any]:
    return {
        "interventions_json": json.dumps([asdict(p.HardIntervention(1, (0,), (4.0,)))]),
        "mode": "matched",
        "population": "9007199254740993",
        "max_seconds": 30,
        "max_bytes": 2_000_000,
    }


def test_session_causal_pair_reopen_traces_and_download(tmp_path: Path) -> None:
    session = ExplorerSession(tmp_path)
    baseline = original(session)
    controls = session.request("intervention_controls", {})
    assert len(controls["nodes"]) == 3
    created = session.request(
        "intervene", {**change(), "expected_episode_id": baseline["view"]["episode_id"]}
    )
    assert created["view"] == baseline["view"]
    assert created["recipe_json"] == baseline["recipe_json"]
    pair = created["intervention"]
    assert all(row["inputs"][1] == 4 for row in pair["after"]["rows"])
    assert pair["card"]["row_identity"]["task_id"] == "9007199254740999"
    assert pair["card"]["row_identity"]["seed"] == "9007199254740997"
    graphs = session.request(
        "intervention_graph",
        {"indices": [15, 0, 15], "expected_intervention_id": pair["after_id"], "max_seconds": 30},
    )
    assert graphs["before"]["rows"] == [15, 0, 15]
    assert graphs["after"]["rows"] == [15, 0, 15]
    assert any(n["id"] == "intervention-1-state" for n in graphs["after"]["nodes"])
    exported = session.request(
        "intervention_export",
        {"expected_intervention_id": pair["after_id"], "max_seconds": 30, "max_bytes": 2_000_000},
    )
    destination = tmp_path / "downloaded"
    with zipfile.ZipFile(io.BytesIO(base64.b64decode(exported["data"]))) as archive:
        assert {"before.zip", "before.json", "after.zip", "after.json", "pair.json"} <= set(
            archive.namelist()
        )
        archive.extractall(destination)
    loaded = load_causal_pair(destination)
    assert json.loads(loaded.before.recipe_json)["task_id"] == 9007199254740999
    assert loaded.after.episode_id == pair["after_id"]
    reopened = ExplorerSession(tmp_path).request(
        "open_intervention", {"path": pair["path"], "max_seconds": 30}
    )
    assert reopened["intervention"] == pair
    assert reopened["recipe_json"] == baseline["recipe_json"]


def test_invalid_timeout_caps_and_stale_actions_preserve_comparison(tmp_path: Path) -> None:
    session = ExplorerSession(tmp_path)
    original(session)
    completed = session.request("intervene", change())
    for override in (
        {"interventions_json": "not-json"},
        {"max_seconds": 0.000001},
        {"max_bytes": 65535},
        {"population": "bad"},
        {"device": "cuda"},
        {"causal_limits": None},
    ):
        with pytest.raises(ValueError, match="failed|CPU|explicit object"):
            session.request("intervene", {**change(), **override})
        assert session.request("view", {}) == completed
    with pytest.raises(ValueError, match="changed"):
        session.request("intervention_export", {"expected_intervention_id": "stale"})
    with pytest.raises(ValueError, match="inside"):
        session.request("open_intervention", {"path": "../outside", "max_seconds": 30})
    with pytest.raises(ValueError, match="failed"):
        session.request(
            "open_intervention",
            {"path": completed["intervention"]["path"], "max_bytes": 65535, "max_seconds": 30},
        )
    assert session.request("view", {}) == completed
    records = [
        json.loads(path.read_text())
        for path in (tmp_path / ".intervention-attempts").glob("*.json")
    ]
    assert any(r["status"] == "FAILED" and r.get("result", {}).get("timed_out") for r in records)
    assert any(
        r["status"] == "FAILED" and r["data"].get("causal_limits", {}) is None for r in records
    )


def test_native_controls_and_forced_target_roundtrip(tmp_path: Path) -> None:
    session = ExplorerSession(tmp_path)
    controls = session.request("construction_examples", {})["examples"][
        "Native mixed graph (authored)"
    ]
    controls["task"].update(rows=16, context_rows=8)
    original_view = session.request(
        "construct",
        {"controls": controls, "construction_seed": 0, "seed": 7, "task_id": 3, "max_seconds": 30},
    )
    info = session.request("intervention_controls", {})
    target = original_view["view"]["recipe"]["recipe"]["target"][0]
    label = info["nodes"][target]["domain"]["vocabulary"][-1]
    changed = session.request(
        "intervene",
        {
            **change(),
            "interventions_json": json.dumps([asdict(p.HardIntervention(target, category=label))]),
        },
    )
    assert changed["view"] == original_view["view"]
    assert changed["intervention"]["after"]["target_emission"] == "forced-category indicator"
    assert all(row["target_label"] == label for row in changed["intervention"]["after"]["rows"])


def test_aggregate_reopen_cap_rejects_before_publishing(tmp_path: Path) -> None:
    session = ExplorerSession(tmp_path)
    original(session)
    completed = session.request("intervene", change())
    # Retained manifest is deliberately padded as a locally produced adversarial fixture.
    # Its contents remain valid JSON; the aggregate cap must run before full replay.
    manifest = tmp_path / completed["intervention"]["path"] / "pair.json"
    manifest.write_text(manifest.read_text() + " " * 70000)
    with pytest.raises(ValueError, match="input storage limit"):
        session.request(
            "open_intervention",
            {"path": completed["intervention"]["path"], "max_seconds": 30, "max_bytes": 65536},
        )
    assert session.request("view", {}) == completed
