"""Sampled fitted parameters remain visible and replayable in shared explorer transport."""

import json
from pathlib import Path

import pytest

from r0.modules.explorer import ExplorerSession


@pytest.mark.parametrize(
    "label",
    [
        "TabICL depth/count laws: dt independent fit (r0)",
        "TabICL depth/count laws: extra_trees independent fit (r0)",
        "TabICL depth/count laws: rf independent fit (r0)",
        "Sampled fitted parameters: gbdt classification (r0)",
    ],
)
def test_sampled_parameter_workbench_frozen_records_and_edits(tmp_path: Path, label: str):
    session = ExplorerSession(tmp_path)
    controls = session.request("construction_examples", {})["examples"][label]
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
    details = result["view"]["construction_details"]
    assert len(details["parameter_samples"]) == 2
    assert all(
        s["lifecycle"] == "independent-calibration-r0-v1" for s in details["parameter_samples"]
    )
    assert all(s["raw_draw_count"] >= 2 for s in details["parameter_samples"])
    if "TabICL" in label:
        assert "floor" in details["parameter_samples"][0]["pinned_formula"]
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
        "open", {"archive": result["archive"], "task_id": "9007199254740993"}
    )
    assert opened["recipe_json"] == result["recipe_json"]
    editor = session.request("mechanism", {})
    assert editor["editable_recipe"]["version"] == "functional-causal-v1"
    before = session.task
    bad = json.loads(json.dumps(controls))
    bad["fits"][0]["prior"]["laws"] = [["rows", {"kind": "fixed", "values": [2048]}]]
    with pytest.raises(ValueError, match="Generation failed"):
        session.request(
            "construct", {"controls": bad, "construction_seed": 2, "seed": 3, "task_id": 4}
        )
    assert session.task is before


def test_sampled_parameters_shared_local_notebook_callback(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
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
    assert 'id="sampled-fitted-controls-note"' in shown[0]
    controls = invoke("construction_examples", {})["application/json"]["examples"][
        "TabICL depth/count laws: dt independent fit (r0)"
    ]
    controls["skeleton"]["task"].update(rows=16, context_rows=8)
    result = invoke(
        "construct", {"controls": controls, "construction_seed": "28", "seed": "7", "task_id": "9"}
    )["application/json"]
    samples = result["view"]["construction_details"]["parameter_samples"]
    assert samples[0]["unused_parameters"] == ["estimators"]
    assert invoke("export", {})["application/json"]["data"]
    opened = invoke("open", {"archive": result["archive"], "task_id": "9"})["application/json"]
    assert opened["recipe_json"] == result["recipe_json"]
