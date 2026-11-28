"""Shared transport keeps frozen XGBoost provenance across task identities."""

import json
from pathlib import Path

import pytest

from r0.modules.explorer import ExplorerSession


@pytest.mark.parametrize("objective", ["reg:squarederror", "binary:logistic", "multi:softprob"])
def test_xgboost_guarded_session(tmp_path: Path, objective: str):
    session = ExplorerSession(tmp_path)
    controls = session.request("construction_examples", {})["examples"][
        f"XGBoost shared-vector fit: {objective}"
    ]
    controls["skeleton"]["task"].update(rows=16, context_rows=8)
    result = session.request(
        "construct",
        {
            "controls": controls,
            "construction_seed": "29",
            "seed": "9007199254740997",
            "task_id": "9007199254740993",
        },
    )
    assert (
        result["view"]["construction_details"]["fits"][0]["actual_plan"]["objective"] == objective
    )
    assert session.request("graph", {"indices": [15, 0, 15]})["graph"]["rows"] == [15, 0, 15]
    advanced = session.request(
        "generate",
        {
            "record_json": result["recipe_json"],
            "frozen_identity": {"seed": "9007199254740997", "task_id": "9007199254740995"},
        },
    )
    assert (
        json.loads(advanced["recipe_json"])["recipe"] == json.loads(result["recipe_json"])["recipe"]
    )
    replay = session.request("generate", {"record_json": result["recipe_json"]})
    assert replay["view"]["episode_id"] == result["view"]["episode_id"]
    assert session.request("export", {})["data"]
    opened = ExplorerSession(tmp_path).request(
        "open",
        {
            "archive": result["archive"],
            "task_id": "9007199254740993",
        },
    )
    assert opened["recipe_json"] == result["recipe_json"]
    assert session.request("mechanism", {})["editable_recipe"]["version"] == "functional-causal-v1"
    before = session.task
    controls["fits"][0]["plan"]["depth"] = 99
    with pytest.raises(ValueError, match="Generation failed"):
        session.request(
            "construct", {"controls": controls, "construction_seed": 2, "seed": 3, "task_id": 4}
        )
    assert session.task is before


def test_xgboost_shared_local_notebook_callback(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    import sys
    from types import ModuleType
    from typing import Any

    from r0.modules.explorer import show_colab_explorer

    callbacks: dict[str, Any] = {}
    shown: list[str] = []
    output = ModuleType("google.colab.output")
    display = ModuleType("IPython.display")

    def register(name: str, callback: Any) -> None:
        callbacks[name] = callback

    def wrap(value: Any) -> dict[str, Any]:
        return {"application/json": value}

    monkeypatch.setattr(output, "register_callback", register, raising=False)
    monkeypatch.setattr(display, "JSON", wrap, raising=False)
    monkeypatch.setattr(display, "HTML", str, raising=False)
    monkeypatch.setattr(display, "display", shown.append, raising=False)
    monkeypatch.setitem(sys.modules, "google.colab.output", output)
    monkeypatch.setitem(sys.modules, "IPython.display", display)
    name = show_colab_explorer(tmp_path)
    invoke = callbacks[name]
    assert 'id="xgboost-controls-note"' in shown[0]
    controls = invoke("construction_examples", {})["application/json"]["examples"][
        "XGBoost shared-vector fit: multi:softprob"
    ]
    controls["skeleton"]["task"].update(rows=16, context_rows=8)
    result = invoke(
        "construct", {"controls": controls, "construction_seed": "28", "seed": "7", "task_id": "9"}
    )["application/json"]
    fits = result["view"]["construction_details"]["fits"]
    assert fits[0]["actual_plan"]["objective"] == "multi:softprob"
    assert invoke("export", {})["application/json"]["data"]
    opened = invoke("open", {"archive": result["archive"], "task_id": "9"})["application/json"]
    assert opened["recipe_json"] == result["recipe_json"]
