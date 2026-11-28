"""Guarded observation workflows keep source receipts and active baselines intact."""

import json
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from r0.modules import generation as g
from r0.modules import priors as p
from r0.modules.explorer import ExplorerSession


def examples(session: ExplorerSession) -> dict[str, Any]:
    return {
        k: v
        for k, v in session.request("construction_examples", {})["examples"].items()
        if k.startswith("Observations:")
    }


def construct(session: ExplorerSession) -> dict[str, Any]:
    controls = examples(session)["Observations: P1 pinned 778f0d0 / dense"]
    return session.request(
        "construct",
        {"controls": controls, "construction_seed": "7", "seed": "37", "task_id": str(2**54 + 17)},
    )


def test_broad_presets_build_full_receipts_and_traces(tmp_path: Path) -> None:
    catalog = examples(ExplorerSession(tmp_path))
    catalog = {k: v for k, v in catalog.items() if "r0 general /" not in k}
    assert len(catalog) == 16
    for label, controls in catalog.items():
        recipe = g.GraphConstruction(json.dumps(controls), 7).build(p.CausalLimits())
        assert isinstance(recipe, p.NetworkObservationRecipe), label
        assert p.recipe_from_dict(asdict(recipe)) == recipe
        task = p.generate_task(recipe, seed=37, task_id=2**54 + 17)
        assert p.explain_generation(task, [31, 0, 31])["signal_max_error"] == 0
        assert task.inputs.shape == (32, 2)
        if "authored affine" in label:
            source = recipe.observations.request.source
            assert isinstance(source, p.NetworkRecipe)
            x = p.generate_task(source, seed=37, task_id=2**54 + 17).inputs.numpy()
            first = np.column_stack(
                (x[:, 0], x[:, 1], x[:, 0] + x[:, 1], x[:, 0] - x[:, 1], 2 * x[:, 0], 2 * x[:, 1])
            )
            second = 2 * first + 1
            literal = np.concatenate((second, 2 * second + 2), axis=1)
            np.testing.assert_allclose(
                task.inputs.numpy(), literal[:, recipe.observations.features], rtol=0, atol=1e-12
            )
            np.testing.assert_allclose(
                task.scores.numpy(), literal[:, recipe.observations.targets], rtol=0, atol=1e-12
            )


def test_guarded_observation_editor_archive_next_and_intervention(tmp_path: Path) -> None:
    session = ExplorerSession(tmp_path)
    original = construct(session)
    record = json.loads(original["recipe_json"])
    assert original["view"]["recipe"]["task_id"] == str(2**54 + 17)
    assert original["view"]["construction_details"]["receipt"] == record["recipe"]["observations"]
    editor = session.request("mechanism", {})
    assert editor["observation_selection"]["profile"] == "p1-778f0d0-v1"
    graph = session.request("graph", {"indices": [31, 0, 31]})["graph"]
    assert graph["rows"] == [31, 0, 31]
    controls = session.request("intervention_controls", {})
    assert controls["observation_selection"] == editor["observation_selection"]
    runtime = editor["editable_recipe"]
    ref = runtime["target"][0]
    pair = session.request(
        "intervene",
        {
            "interventions_json": json.dumps(
                [asdict(p.HardIntervention(ref[0], channels=(ref[1],), values=(2.5,)))]
            ),
            "mode": "matched",
        },
    )
    assert pair["recipe_json"] == original["recipe_json"]
    compared = session.request("pair", {"intervention": "noise", "value": 0.5})
    assert compared["recipe_json"] == original["recipe_json"]
    assert compared["pair"]["card"]["observed_equalities"]["scores"]
    before = session.request("view", {})
    invalid = deepcopy(runtime)
    invalid["nodes"][1]["function"] = {}
    with pytest.raises(ValueError, match="failed"):
        session.request("observation_draft", {"recipe": invalid})
    assert session.request("view", {}) == before
    prepared = session.request("observation_draft", {"recipe": runtime})
    edited = session.request("edit", {"recipe": prepared["editable_recipe"]})
    assert edited["recipe_json"] == original["recipe_json"]
    reopened = session.request("open_edit", {"path": edited["edit"]["path"]})
    assert reopened["recipe_json"] == original["recipe_json"]
    next_task = session.request(
        "generate",
        {
            "record_json": original["recipe_json"],
            "frozen_identity": {"seed": "37", "task_id": str(2**54 + 18)},
        },
    )
    assert json.loads(next_task["recipe_json"])["recipe"] == record["recipe"]
    replay = session.request("generate", {"record_json": original["recipe_json"]})
    assert replay["recipe_json"] == original["recipe_json"]
    assert (
        session.request("open", {"archive": original["archive"], "task_id": str(2**54 + 17)})[
            "recipe_json"
        ]
        == original["recipe_json"]
    )
    assert session.request("export", {})["data"]
    curve = session.request("curve", {"feature": 0, "points": 3})["curve"]
    assert curve["kind"] == "observation_causal_reference"
    assert curve["recipe_json"] == original["recipe_json"]


def test_guarded_observation_decode_and_timeout_preserve_baseline(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    session = ExplorerSession(tmp_path)
    baseline = construct(session)
    recipe = p.recipe_from_dict(json.loads(baseline["recipe_json"])["recipe"])
    assert isinstance(recipe, p.NetworkObservationRecipe)

    def forbidden(*args: object, **kwargs: object):
        raise AssertionError("construction snapshot replay ran in parent process")

    monkeypatch.setattr(type(recipe.observations.initial), "restore", forbidden)
    editor = session.request("mechanism", {})
    session.request("intervention_controls", {})
    session.request("graph", {"indices": [31, 0, 31]})
    for action, data in (
        ("mechanism", {}),
        ("intervention_controls", {}),
        ("graph", {"indices": [0]}),
        ("observation_draft", {"recipe": editor["editable_recipe"]}),
    ):
        with pytest.raises(ValueError, match="timed out"):
            session.request(action, {**data, "max_seconds": 0.001})
        assert session.request("view", {}) == baseline


@pytest.mark.parametrize("problem", ["skip", "width", "policy", "seed"])
def test_invalid_observation_controls_preserve_saved_task(tmp_path: Path, problem: str) -> None:
    session = ExplorerSession(tmp_path)
    baseline = construct(session)
    controls = examples(session)["Observations: P1 pinned 778f0d0 / dense"]
    if problem == "skip":
        controls["source"]["network_prior"]["families"] = ["skip"]
    elif problem == "width":
        controls["source"]["network_prior"]["widths"] = [6, 5, 6]
    elif problem == "policy":
        controls["observations"]["selection"] = "unknown"
    else:
        controls["selection_seed"] = 1.5
    with pytest.raises(ValueError, match="failed|Failed"):
        session.request(
            "construct",
            {"controls": controls, "construction_seed": 7, "seed": 37, "task_id": 2**54 + 17},
        )
    assert session.request("view", {}) == baseline


def test_shared_local_notebook_observation_callback(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import sys
    from types import ModuleType

    from r0.modules.explorer import show_colab_explorer

    callbacks: dict[str, Any] = {}
    shown: list[str] = []
    output, display = ModuleType("google.colab.output"), ModuleType("IPython.display")

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
    invoke = callbacks[show_colab_explorer(tmp_path)]
    assert 'id="observation-controls"' in shown[0]
    controls = invoke("construction_examples", {})["application/json"]["examples"][
        "Observations: authored affine equations / pinned selection"
    ]
    controls["task"].update(task_type="multiclass", classes=3)
    result = invoke(
        "construct",
        {"controls": controls, "construction_seed": "7", "seed": "37", "task_id": str(2**54 + 17)},
    )["application/json"]
    assert result["view"]["recipe"]["task_id"] == str(2**54 + 17)
    assert len(result["view"]["classes"]) == 3
    assert invoke("mechanism", {})["application/json"]["observation_selection"]
    assert invoke("graph", {"indices": [31, 0, 31]})["application/json"]["graph"]["rows"] == [
        31,
        0,
        31,
    ]
    assert invoke("export", {})["application/json"]["data"]
    assert (
        invoke("open", {"archive": result["archive"], "task_id": str(2**54 + 17)})[
            "application/json"
        ]["recipe_json"]
        == result["recipe_json"]
    )
