"""Network input views remain frozen and editable through the public workbench."""

import json
from dataclasses import asdict, replace
from pathlib import Path

import torch

from r0.modules import priors as p
from r0.modules.explorer import ExplorerSession


def open_view(path: Path):
    task = p.LinearRecipe(rows=16, context_rows=8, features=2, signal_features=2, active_features=2)
    fn = p.FrozenNetworkFunction(
        2, (p.NetworkBlock((0,), (p.NetworkAffine(((2.0, 3.0),), (0.1,)),)),)
    )
    view = p.NetworkInputView(
        p.NetworkRecipe(task, fn), replace(task, features=4, active_features=1), (1,)
    )
    session = ExplorerSession(path)
    result = session.request(
        "generate", {"recipe": asdict(view), "seed": 9, "task_id": "9007199254740993"}
    )
    return session, result


def test_view_construction_details_retain_input_mask(tmp_path: Path):
    _, result = open_view(tmp_path)
    details = result["view"]["construction_details"]
    assert details["input_view"]["active_inputs"] == [1]
    assert details["input_view"]["observed_features"] == 4
    assert details["input_width"] == 2


def test_frozen_view_next_retains_complete_mechanism(tmp_path: Path):
    session, before = open_view(tmp_path)
    after = session.request(
        "generate",
        {
            "record_json": before["recipe_json"],
            "frozen_identity": {"seed": "9", "task_id": "9007199254740995"},
        },
    )
    assert json.loads(after["recipe_json"])["recipe"] == json.loads(before["recipe_json"])["recipe"]
    assert json.loads(after["recipe_json"])["task_id"] == 9007199254740995


def test_view_architecture_edit_retains_observation_mask(tmp_path: Path):
    session, before = open_view(tmp_path)
    original = session.task
    assert original is not None
    editor = session.request("mechanism", {})
    draft = json.loads(json.dumps(editor["editable_recipe"]))
    assert draft["active_inputs"] == [1]
    draft["base"]["network"]["blocks"][0]["experts"][0]["bias"][0] += 2
    validated = session.request("network_draft", {"recipe": draft})["recipe"]
    result = session.request("edit", {"recipe": validated})
    assert result["recipe_json"] == before["recipe_json"]
    reopened = ExplorerSession(tmp_path).request("open_edit", {"path": result["edit"]["path"]})
    after_raw = reopened["edit"]["after"]["recipe"]["recipe"]
    assert after_raw["active_inputs"] == [1]
    edited = p.generate_task(p.recipe_from_dict(after_raw), seed=9, task_id=9007199254740993)
    torch.testing.assert_close(edited.inputs, original.inputs, rtol=0, atol=0)
    torch.testing.assert_close(edited.scores, original.scores + 2, rtol=1e-14, atol=1e-14)
