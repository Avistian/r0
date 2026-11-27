"""Shared bounded explorer transport preserves explicit fitting lifecycle archives."""

import json
from pathlib import Path

import pytest
import torch

from r0.modules.explorer import ExplorerSession
from r0.modules.generation import GraphConstruction


@pytest.mark.parametrize("mode", ["support-only", "batch-refit-reference", "mixed-independent"])
def test_population_workbench_replay_edit_export(tmp_path: Path, mode: str):
    session = ExplorerSession(tmp_path)
    label = f"Population fit: {mode} dt regression (r0)"
    controls = session.request("construction_examples", {})["examples"][label]
    controls["skeleton"]["task"].update(rows=16, context_rows=4)
    result = session.request(
        "construct",
        {
            "controls": controls,
            "construction_seed": "11",
            "seed": "9007199254740997",
            "task_id": "9007199254740993",
        },
    )
    details = result["view"]["construction_details"]
    assert details["population"] == mode
    assert details["fit_task_id"] == "9007199254740993"
    assert details["fits"][0]["actual_plan"]["rows"] == (
        4 if mode == "support-only" else 16 if mode == "batch-refit-reference" else 64
    )
    before = session.task
    assert session.request("graph", {"indices": [15, 0, 15]})["graph"]["rows"] == [15, 0, 15]
    assert session.request("export", {})["data"]
    opened = ExplorerSession(tmp_path).request(
        "open", {"archive": result["archive"], "task_id": "9007199254740993"}
    )
    assert opened["recipe_json"] == result["recipe_json"]

    editor = session.request("mechanism", {})
    assert editor["editable_recipe"]["version"] == (
        "support-population-functional-causal-v1"
        if mode == "support-only"
        else "functional-causal-v1"
    )
    from r0.modules import priors

    unchanged = priors.generate_task(
        priors.recipe_from_dict(editor["editable_recipe"]),
        seed=9007199254740997,
        task_id=9007199254740993,
    )
    assert before is not None
    assert torch.equal(unchanged.inputs, before.inputs)
    assert torch.equal(unchanged.scores, before.scores)
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
    baseline = session.task
    controls["fits"][0]["plan"]["rows"] = 2000
    with pytest.raises(ValueError, match="Generation failed"):
        session.request(
            "construct", {"controls": controls, "construction_seed": 1, "seed": 1, "task_id": 1}
        )
    assert session.task is baseline


def test_population_shared_local_notebook_callback(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
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
    assert 'id="population-fitted-controls-note"' in shown[0]
    controls = invoke("construction_examples", {})["application/json"]["examples"][
        "Population fit: support-only dt regression (r0)"
    ]
    controls["skeleton"]["task"].update(rows=16, context_rows=8)
    result = invoke(
        "construct", {"controls": controls, "construction_seed": "28", "seed": "7", "task_id": "9"}
    )["application/json"]
    fits = result["view"]["construction_details"]["fits"]
    assert fits[0]["actual_plan"]["rows"] == 8
    assert invoke("export", {})["application/json"]["data"]
    opened = invoke("open", {"archive": result["archive"], "task_id": "9"})["application/json"]
    assert opened["recipe_json"] == result["recipe_json"]


def test_new_batch_construction_redraws_targets_while_replay_is_frozen(tmp_path: Path):
    from r0.modules import priors

    controls = ExplorerSession(tmp_path).request("construction_examples", {})["examples"][
        "Population fit: batch-refit-reference dt regression (r0)"
    ]
    controls["skeleton"]["task"].update(rows=16, context_rows=8)
    construction = GraphConstruction(json.dumps(controls), 7)
    a = construction.build(priors.CausalLimits(), episode_seed=3, task_id=11)
    b = construction.build(priors.CausalLimits(), episode_seed=3, task_id=12)
    assert isinstance(a, priors.PopulationFittedGraph)
    assert isinstance(b, priors.PopulationFittedGraph)
    assert a.fits[0].realization.targets != b.fits[0].realization.targets
    assert construction.build(priors.CausalLimits(), episode_seed=3, task_id=11) == a
