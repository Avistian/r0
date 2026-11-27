"""Standalone architecture controls retain frozen evidence across shared explorer paths."""

import base64
import io
import json
import zipfile
from copy import deepcopy
from pathlib import Path
from typing import Any

import pytest

from r0.modules.explorer import ExplorerSession


def controls(session: ExplorerSession, label: str) -> dict[str, Any]:
    result = session.request("construction_examples", {})["examples"][label]
    result["task"].update(rows=16, context_rows=8)
    return json.loads(json.dumps(result))


def construct(session: ExplorerSession, control: dict[str, Any], **extra: Any) -> dict[str, Any]:
    return session.request(
        "construct",
        {
            "controls": control,
            "construction_seed": "9007199254740997",
            "seed": "9007199254741011",
            "task_id": "9007199254740993",
            **extra,
        },
    )


@pytest.mark.parametrize(
    "family",
    ["dense", "individual-sparse", "block-sparse", "skip", "residual", "irregular", "gated"],
)
def test_network_family_construct_trace_frozen_replay_export(tmp_path: Path, family: str):
    session = ExplorerSession(tmp_path)
    result = construct(session, controls(session, f"Standalone network: {family}"))
    raw = json.loads(result["recipe_json"])
    original = deepcopy(raw["recipe"]["network"])
    assert result["view"]["construction_details"]["kind"] == "standalone-network"
    assert result["view"]["construction_details"]["raw_draw_count"] > 0
    graph = session.request("graph", {"indices": [15, 0, 15]})["graph"]
    assert graph["rows"] == [15, 0, 15]
    assert any("expert" in n["id"] for n in graph["nodes"])
    regenerated = session.request(
        "generate",
        {
            "record_json": result["recipe_json"],
            "frozen_identity": {"seed": "9007199254741011", "task_id": "9007199254740995"},
        },
    )
    after = json.loads(regenerated["recipe_json"])
    assert after["task_id"] == 9007199254740995
    assert after["recipe"]["network"] == original
    replay = session.request("generate", {"record_json": result["recipe_json"]})
    assert replay["view"]["episode_id"] == result["view"]["episode_id"]
    bundle = session.request("export", {})
    with zipfile.ZipFile(io.BytesIO(base64.b64decode(bundle["data"]))) as archive:
        assert archive.testzip() is None
        assert "predictor.csv" in archive.namelist()
    reopened = ExplorerSession(tmp_path).request(
        "open", {"archive": result["archive"], "task_id": "9007199254740993"}
    )
    assert reopened["recipe_json"] == result["recipe_json"]


@pytest.mark.parametrize(
    "label",
    [
        "Standalone network: p1-individual source laws",
        "Standalone network: p1-block source laws",
        "Standalone network: sparse + gated + skip (authored)",
    ],
)
def test_source_profiles_and_authored_combinations_edit_save_reopen(tmp_path: Path, label: str):
    session = ExplorerSession(tmp_path)
    result = construct(session, controls(session, label))
    editor = session.request("mechanism", {})
    assert editor["network"]
    assert editor["editable_recipe"]["network"]["version"] == "frozen-structural-network-v1"
    draft = json.loads(json.dumps(editor["editable_recipe"]))
    draft["network"]["blocks"][-1]["experts"][0]["bias"][0] += 2.0
    edited = session.request("edit", {"recipe": draft})
    assert edited["view"]["episode_id"] == result["view"]["episode_id"]
    assert edited["edit"]["card"]["after_episode_id"] != result["view"]["episode_id"]
    assert edited["recipe_json"] == result["recipe_json"]
    assert session.request("edit_export", {})["data"]
    reopened = ExplorerSession(tmp_path).request("open_edit", {"path": edited["edit"]["path"]})
    assert reopened["edit"]["card"] == edited["edit"]["card"]


def test_invalid_and_timedout_network_actions_preserve_active_baseline(tmp_path: Path):
    session = ExplorerSession(tmp_path)
    baseline = construct(session, controls(session, "Standalone network: gated"))
    bad = controls(session, "Standalone network: dense")
    bad["network_prior"]["widths"][-1] = 2
    with pytest.raises(ValueError, match="Generation failed"):
        construct(session, bad)
    assert session.request("view", {}) == baseline
    with pytest.raises(ValueError, match="Generation failed"):
        construct(session, controls(session, "Standalone network: residual"), max_seconds=1e-9)
    assert session.request("view", {}) == baseline
    editor = session.request("mechanism", {})
    draft = json.loads(json.dumps(editor["editable_recipe"]))
    draft["network"]["blocks"][0]["parents"] = [99]
    with pytest.raises(ValueError, match="earlier"):
        session.request("edit", {"recipe": draft})
    assert session.request("view", {}) == baseline
    with pytest.raises(ValueError, match="Generation failed"):
        session.request("edit", {"recipe": editor["editable_recipe"], "max_seconds": 1e-9})
    assert session.request("view", {}) == baseline
    failed = list((tmp_path / ".attempts").glob("*/*-result.json"))
    assert sum(json.loads(p.read_text())["status"] == "FAILED" for p in failed) >= 3


def test_network_colab_callback_uses_same_construction_edit_and_export(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    import sys
    from types import ModuleType

    from r0.modules.explorer import show_colab_explorer

    callbacks: dict[str, Any] = {}
    displayed: list[str] = []
    output = ModuleType("google.colab.output")

    def register(name: str, callback: Any) -> None:
        callbacks[name] = callback

    monkeypatch.setattr(output, "register_callback", register, raising=False)
    display = ModuleType("IPython.display")

    def wrap(value: Any) -> dict[str, Any]:
        return {"application/json": value}

    monkeypatch.setattr(display, "JSON", wrap, raising=False)
    monkeypatch.setattr(display, "HTML", str, raising=False)
    monkeypatch.setattr(display, "display", displayed.append, raising=False)
    monkeypatch.setitem(sys.modules, "google.colab.output", output)
    monkeypatch.setitem(sys.modules, "IPython.display", display)
    name = show_colab_explorer(tmp_path)
    invoke = callbacks[name]
    assert 'id="network-workbench-content"' in displayed[0]
    examples = invoke("construction_examples", {})["application/json"]["examples"]
    control = examples["Standalone network: sparse + gated + skip (authored)"]
    control["task"].update(rows=16, context_rows=8)
    result = invoke(
        "construct", {"controls": control, "construction_seed": "12", "seed": "7", "task_id": "9"}
    )["application/json"]
    assert result["view"]["construction_details"]["kind"] == "standalone-network"
    editor = json.loads(json.dumps(invoke("mechanism", {})["application/json"]))
    editor["editable_recipe"]["network"]["blocks"][-1]["experts"][0]["bias"][0] += 1
    edit = invoke("edit", {"recipe": editor["editable_recipe"]})["application/json"]
    assert (
        invoke("open_edit", {"path": edit["edit"]["path"]})["application/json"]["edit"]["card"]
        == edit["edit"]["card"]
    )
    assert invoke("edit_export", {})["application/json"]["data"]
