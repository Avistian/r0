"""Source scalar laws use the guarded local/notebook network workbench."""

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import pytest

from r0.modules import priors as p
from r0.modules.explorer import ExplorerSession


def controls(source: str = "p1-paper", scope: str = "shared") -> dict[str, Any]:
    request = p.network_source_hyperprior(
        p.NetworkSamplingRequest(2, (1,), families=("dense",)), source=source
    )
    raw = asdict(request)
    raw["depth"] = asdict(p.NetworkScalarPrior("fixed", 2.0, 2.0))
    raw["width"] = asdict(p.NetworkScalarPrior("integer-uniform", 3.0, 6.0))
    raw["width_scope"] = scope
    raw["template"].pop("limits")
    task = p.LinearRecipe(rows=19, context_rows=9, features=2, signal_features=2, active_features=2)
    return {"topology": "network-hyperprior", "task": asdict(task), "root": None, "hyperprior": raw}


def test_hyperprior_presets_keep_source_laws_and_width_options(tmp_path: Path):
    examples = ExplorerSession(tmp_path).request("construction_examples", {})["examples"]
    profiles = [v for v in examples.values() if v.get("topology") == "network-hyperprior"]
    assert len(profiles) == 28
    assert {v["hyperprior"]["width_scope"] for v in profiles} == {"shared", "per-hidden-block"}
    assert {v["hyperprior"]["depth"]["family"] for v in profiles} == {
        "p1-paper-tnlu-v1",
        "tabicl-9530bd6-tnlu-v1",
    }


@pytest.mark.parametrize("source", ["p1-paper", "tabicl-9530bd6"])
def test_profile_construct_frozen_replay_edit_and_limits(tmp_path: Path, source: str):
    session = ExplorerSession(tmp_path)
    result = session.request(
        "construct",
        {
            "controls": controls(source, "per-hidden-block"),
            "construction_seed": "7",
            "seed": "11",
            "task_id": str(2**54 + 3),
        },
    )
    raw = json.loads(result["recipe_json"])["recipe"]
    assert raw["network"]["version"] == "sampled-hyperprior-network-v1"
    details = result["view"]["construction_details"]
    assert details["hyperprior"]["profile"]["width_scope"] == "per-hidden-block"
    assert details["stream_roles"] == ["hyper", "named", "raw"]
    editor = session.request("mechanism", {})
    assert editor["editable_recipe"]["network"]["version"] == "frozen-structural-network-v1"
    next_result = session.request(
        "generate",
        {
            "record_json": result["recipe_json"],
            "frozen_identity": {"seed": "11", "task_id": str(2**54 + 5)},
        },
    )
    assert json.loads(next_result["recipe_json"])["recipe"] == raw
    baseline = session.request("view", {})
    bad = controls(source)
    bad["hyperprior"]["depth"] = asdict(p.NetworkScalarPrior("fixed", 100.0, 100.0))
    with pytest.raises(ValueError, match="Generation failed"):
        session.request(
            "construct", {"controls": bad, "construction_seed": 7, "seed": 11, "task_id": 1}
        )
    assert session.request("view", {}) == baseline
    assert any("operational cap" in path.read_text() for path in tmp_path.glob("**/attempt-*.json"))
    with pytest.raises(ValueError, match="Generation failed"):
        session.request(
            "construct",
            {
                "controls": controls(source),
                "construction_seed": 7,
                "seed": 11,
                "task_id": 1,
                "causal_limits": {"parameters": 100},
            },
        )
    assert session.request("view", {}) == baseline


def test_profile_local_notebook_callback_edit_export(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    import sys
    from types import ModuleType

    from r0.modules.explorer import show_colab_explorer

    callbacks: dict[str, Any] = {}
    output = ModuleType("google.colab.output")
    display = ModuleType("IPython.display")

    def register(name: str, callback: Any) -> None:
        callbacks[name] = callback

    def wrap(value: Any) -> dict[str, Any]:
        return {"application/json": value}

    def discard(value: Any) -> None:
        pass

    monkeypatch.setattr(output, "register_callback", register, raising=False)
    monkeypatch.setattr(display, "JSON", wrap, raising=False)
    monkeypatch.setattr(display, "HTML", str, raising=False)
    monkeypatch.setattr(display, "display", discard, raising=False)
    monkeypatch.setitem(sys.modules, "google.colab.output", output)
    monkeypatch.setitem(sys.modules, "IPython.display", display)
    callback = callbacks[show_colab_explorer(tmp_path)]
    result = callback(
        "construct",
        {
            "controls": controls("tabicl-9530bd6"),
            "construction_seed": "7",
            "seed": "11",
            "task_id": str(2**54 + 3),
        },
    )["application/json"]
    assert result["view"]["construction_details"]["hyperprior"]["parameters"]
    editor = callback("mechanism", {})["application/json"]
    edited = json.loads(json.dumps(editor["editable_recipe"]))
    edited["network"]["blocks"][-1]["experts"][0]["bias"][0] += 0.5
    changed = callback("edit", {"recipe": edited})["application/json"]
    assert changed["recipe_json"] == result["recipe_json"]
    opened = callback("open_edit", {"path": changed["edit"]["path"]})["application/json"]
    assert (
        opened["edit"]["after"]["recipe"]["recipe"]["network"]["version"]
        == "frozen-structural-network-v1"
    )
    exported = callback("export", {})["application/json"]
    assert exported["data"]
