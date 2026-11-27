"""Guarded prototype workbench retains independent banks and authored transitions."""

import copy
import json
from pathlib import Path

import pytest

from r0.modules.explorer import ExplorerSession

PREFIX = "Prototype root: dense / "


def controls(session: ExplorerSession, mode: str = "multinomial-trial-temperature-r0"):
    examples = session.request("construction_examples", {})["examples"]
    value = examples[PREFIX + mode + " / ceil-r0"]
    value["source"]["source"]["task"].update(rows=16, context_rows=8)
    return value


def test_prototype_presets_cover_architectures_weight_laws_and_count_conventions(tmp_path: Path):
    examples = ExplorerSession(tmp_path).request("construction_examples", {})["examples"]
    selected = [v for v in examples.values() if v.get("topology") == "prototype-root-network"]
    assert len(selected) == 63
    assert len({v["source"]["source"]["network_prior"]["families"][0] for v in selected}) == 7
    assert len({v["weights"]["mode"] for v in selected}) == 3
    assert {v["count_rule"] for v in selected} == {"ceil-r0", "floor-r0", "nearest-even-r0"}


@pytest.mark.parametrize(
    "mode",
    [
        "multinomial-trial-temperature-r0",
        "multinomial-probability-temperature-r0",
        "dirichlet-concentration-r0",
    ],
)
def test_bank_construction_next_replay_trace_edit_and_reopen(mode: str, tmp_path: Path):
    session = ExplorerSession(tmp_path)
    options = controls(session, mode)
    options["bank_seed"] = str(2**54 + 1)
    options["weights"]["probabilities"] = [0.1, 0.2, 0.3, 0.4]
    first = session.request(
        "construct",
        {"controls": options, "construction_seed": "7", "seed": "11", "task_id": str(2**54 + 3)},
    )
    root = json.loads(first["recipe_json"])["recipe"]["root"]
    assert root["version"] == "prototype-root-base-v1"
    assert root["bank"]["root"]["weights"]["mode"] == mode
    details = first["view"]["construction_details"]["source_root"]
    assert details["kind"] == "prototype-convex-mixture-root"
    assert details["sampling"] == root["bank"]
    recipe = json.loads(first["recipe_json"])["recipe"]
    next_task = session.request(
        "generate", {"recipe": recipe, "seed": "11", "task_id": str(2**54 + 4)}
    )
    assert json.loads(next_task["recipe_json"])["recipe"]["root"] == root
    session.request("generate", {"record_json": first["recipe_json"]})
    assert session.request("graph", {"indices": [15, 0, 15]})["graph"]["rows"] == [15, 0, 15]
    editor = session.request("mechanism", {})
    draft = copy.deepcopy(editor["editable_recipe"])
    draft["root"]["bank"] = copy.deepcopy(root["bank"]["root"])
    draft["root"]["bank"]["prototypes"][0][0] += 0.125
    valid = session.request("network_draft", {"recipe": draft})
    saved = session.request("edit", valid)
    reopened = ExplorerSession(tmp_path).request("open_edit", {"path": saved["edit"]["path"]})
    assert reopened["edit"]["card"] == saved["edit"]["card"]
    assert session.request("edit_export", {})["data"]
    assert session.request("view", {})["recipe_json"] == first["recipe_json"]


def test_batch_reference_rejects_without_changing_current_archive(tmp_path: Path):
    session = ExplorerSession(tmp_path)
    value = controls(session)
    first = session.request(
        "construct", {"controls": value, "construction_seed": 7, "seed": 11, "task_id": 3}
    )
    value["lifecycle"] = "prototype-batch-reference-v1"
    with pytest.raises(ValueError, match="Generation failed"):
        session.request(
            "construct", {"controls": value, "construction_seed": 7, "seed": 11, "task_id": 3}
        )
    assert session.request("view", {})["recipe_json"] == first["recipe_json"]


def test_prototype_source_normalization_and_internal_noise_compose(tmp_path: Path):
    session = ExplorerSession(tmp_path)
    examples = session.request("construction_examples", {})["examples"]
    options = controls(session, "dirichlet-concentration-r0")
    options["source"].update(modes=["standardize", "center"], calibration_rows=32)
    noise = examples["Noisy network: dense"]
    noise["source"] = options
    first = session.request(
        "construct", {"controls": noise, "construction_seed": 7, "seed": 11, "task_id": 3}
    )
    recipe = json.loads(first["recipe_json"])["recipe"]
    assert recipe["source"]["root"]["version"] == "prototype-root-base-v1"
    assert recipe["source"]["root"]["bank"]["plan"]["base"]["calibration_rows"] == 32
    assert recipe["nodes"][0]["root"] == recipe["source"]["root"]
    assert session.request("graph", {"indices": [15, 0, 15]})["graph"]["rows"] == [15, 0, 15]
    assert session.request("curve", {"feature": 0, "points": 7})["curve"]["reference"] == "zero"
    replay = session.request("generate", {"record_json": first["recipe_json"]})
    assert replay["recipe_json"] == first["recipe_json"]


def test_prototype_shared_local_notebook_callback(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
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
    assert 'id="prototype-controls"' in shown[0]
    controls = invoke("construction_examples", {})["application/json"]["examples"][
        PREFIX + "dirichlet-concentration-r0 / ceil-r0"
    ]
    controls["source"]["source"]["task"].update(rows=16, context_rows=8)
    result = invoke(
        "construct", {"controls": controls, "construction_seed": "28", "seed": "7", "task_id": "9"}
    )["application/json"]
    assert (
        result["view"]["construction_details"]["source_root"]["kind"]
        == "prototype-convex-mixture-root"
    )
    assert invoke("export", {})["application/json"]["data"]
    opened = invoke("open", {"archive": result["archive"], "task_id": "9"})["application/json"]
    assert opened["recipe_json"] == result["recipe_json"]
