"""Composed law/lifecycle controls retain exact fitted parameter records end to end."""

import json
from pathlib import Path

import pytest

from r0.modules.explorer import ExplorerSession


@pytest.mark.parametrize(
    "population", ["support-only", "batch-refit-reference", "mixed-independent"]
)
@pytest.mark.parametrize("profile", ["explicit laws", "pinned depth/count"])
def test_sampled_population_transport_and_row_contract(
    tmp_path: Path,
    population: str,
    profile: str,
):
    session = ExplorerSession(tmp_path)
    label = f"Sampled population: {population} dt regression, {profile} (r0)"
    examples = session.request("construction_examples", {})["examples"]
    assert len([name for name in examples if name.startswith("Sampled population:")]) == 54
    controls = examples[label]
    controls["skeleton"]["task"].update(rows=16, context_rows=1)
    result = session.request(
        "construct",
        {
            "controls": controls,
            "construction_seed": "28",
            "seed": "9007199254740997",
            "task_id": "9007199254740993",
        },
    )
    details = result["view"]["construction_details"]
    assert details["population"] == population
    assert len(details["parameter_samples"]) == len(details["fits"]) == 2
    for sample, fit in zip(details["parameter_samples"], details["fits"], strict=True):
        assert sample["actual_plan"] == fit["actual_plan"]
        if population != "mixed-independent":
            assert sample["actual_plan"]["rows"] == (1 if population == "support-only" else 16)
        if profile == "pinned depth/count":
            assert sample["unused_parameters"] == ["estimators"]
    assert session.request("graph", {"indices": [15, 0, 15]})["graph"]["rows"] == [15, 0, 15]
    advanced = session.request(
        "generate",
        {
            "record_json": result["recipe_json"],
            "frozen_identity": {"seed": "9007199254740997", "task_id": "9007199254740995"},
        },
    )
    assert (
        json.loads(result["recipe_json"])["recipe"] == json.loads(advanced["recipe_json"])["recipe"]
    )
    session.request("generate", {"record_json": result["recipe_json"]})
    assert session.request("export", {})["data"]
    reopened = ExplorerSession(tmp_path).request(
        "open", {"archive": result["archive"], "task_id": "9007199254740993"}
    )
    assert reopened["recipe_json"] == result["recipe_json"]
    editor = session.request("mechanism", {})
    assert "parameter_samples" not in editor["editable_recipe"]
    baseline = session.task
    bad = json.loads(json.dumps(controls))
    if population != "mixed-independent":
        bad["fits"][0]["prior"]["laws"].append(["rows", {"values": [1]}])
    else:
        bad["fits"][0]["prior"]["base"]["family"] = "xgboost"
    with pytest.raises(ValueError, match="Generation failed"):
        session.request(
            "construct", {"controls": bad, "construction_seed": 3, "seed": 4, "task_id": 5}
        )
    assert session.task is baseline


def test_sampled_population_shared_local_notebook_callback(
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
    assert 'id="sampled-population-controls-note"' in shown[0]
    controls = invoke("construction_examples", {})["application/json"]["examples"][
        "Sampled population: support-only dt regression, pinned depth/count (r0)"
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
