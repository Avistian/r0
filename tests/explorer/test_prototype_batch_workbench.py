"""Batch workbench reconstructs new populations and replays saved populations exactly."""

import copy
import json
from pathlib import Path
from typing import Any, cast

import pytest

from r0.modules.explorer import ExplorerSession


def controls(session: ExplorerSession, mode: str):
    presets = session.request("construction_examples", {})["examples"]
    value = presets[f"Prototype root: dense / {mode} / ceil-r0"]
    value["source"]["source"]["task"].update(rows=16, context_rows=8)
    value.update(lifecycle="prototype-batch-reference-v1", reference_rows=None, bank_seed=None)
    return value


@pytest.mark.parametrize(
    "mode",
    [
        "multinomial-trial-temperature-r0",
        "multinomial-probability-temperature-r0",
        "dirichlet-concentration-r0",
    ],
)
def test_batch_population_workbench(mode: str, tmp_path: Path):
    session = ExplorerSession(tmp_path)
    first = session.request(
        "construct",
        {
            "controls": controls(session, mode),
            "construction_seed": "7",
            "seed": "11",
            "task_id": str(2**54 + 3),
        },
    )
    recipe = json.loads(first["recipe_json"])["recipe"]
    assert recipe["version"] == "prototype-batch-network-realized-v1"
    assert recipe["population"]["plan"]["reference_rows"] == 16
    assert first["view"]["construction_details"]["kind"] == "prototype-batch-network"
    changed = session.request(
        "generate",
        {
            "record_json": first["recipe_json"],
            "population_identity": {"seed": "11", "task_id": str(2**54 + 4)},
        },
    )
    assert json.loads(changed["recipe_json"])["recipe"]["population"] != recipe["population"]
    replay = session.request("generate", {"record_json": first["recipe_json"]})
    assert replay["recipe_json"] == first["recipe_json"]
    assert session.request("graph", {"indices": [15, 0, 15]})["graph"]["rows"] == [15, 0, 15]
    assert len(session.request("curve", {"feature": 0, "points": 7})["curve"]["scores"]) == 7
    editor = session.request("mechanism", {})
    draft = copy.deepcopy(editor["editable_recipe"])
    assert draft["version"] == "prototype-batch-network-request-v1"
    draft["source"]["network"]["blocks"][-1]["experts"][0]["bias"][0] += 0.125
    valid = session.request("network_draft", {"recipe": draft})
    saved = session.request("edit", valid)
    reopened = ExplorerSession(tmp_path).request("open_edit", {"path": saved["edit"]["path"]})
    assert reopened["edit"]["card"] == saved["edit"]["card"]
    assert session.request("edit_export", {})["data"]
    assert session.request("view", {})["recipe_json"] == first["recipe_json"]


def test_batch_changed_rows_reconstruct_and_contradictions_fail_guarded(tmp_path: Path):
    session = ExplorerSession(tmp_path)
    options = controls(session, "multinomial-trial-temperature-r0")
    first = session.request(
        "construct", {"controls": options, "construction_seed": 7, "seed": 11, "task_id": 3}
    )
    raw = json.loads(first["recipe_json"])["recipe"]
    draft = copy.deepcopy(raw["request"])
    draft["source"]["task"]["rows"] = 32
    larger = session.request("generate", {"recipe": draft, "seed": 11, "task_id": 3})
    realized = json.loads(larger["recipe_json"])["recipe"]
    assert len(realized["population"]["root"]["prototypes"]) == 4
    options["reference_rows"] = 32
    with pytest.raises(ValueError, match="Generation failed"):
        session.request(
            "construct", {"controls": options, "construction_seed": 7, "seed": 11, "task_id": 3}
        )
    assert session.request("view", {})["recipe_json"] == larger["recipe_json"]


def test_batch_task_override_keeps_raw_source_and_rejects_geometry(tmp_path: Path):
    session = ExplorerSession(tmp_path)
    first = session.request(
        "construct",
        {
            "controls": controls(session, "dirichlet-concentration-r0"),
            "construction_seed": 7,
            "seed": 11,
            "task_id": 3,
        },
    )
    raw = json.loads(first["recipe_json"])["recipe"]
    task = copy.deepcopy(raw["request"]["source"]["task"])
    task["rows"] = 32
    payload = {
        "record_json": first["recipe_json"],
        "population_identity": {"seed": "11", "task_id": "3"},
        "population_task": task,
    }
    changed = session.request("generate", payload)
    result = json.loads(changed["recipe_json"])["recipe"]
    assert result["population"]["plan"]["reference_rows"] == 32
    assert json.dumps(result["request"]["source"]["root"]) == json.dumps(
        raw["request"]["source"]["root"]
    )
    task["features"] = 3
    with pytest.raises(ValueError, match="geometry"):
        session.request("generate", payload)
    assert session.request("view", {})["recipe_json"] == changed["recipe_json"]


def test_batch_browser_number_spelling_restores_unchanged_custody(tmp_path: Path):
    def javascript(value: Any) -> Any:
        if type(value) is float and value.is_integer():
            return int(value)
        if type(value) is list:
            return [javascript(v) for v in cast(list[Any], value)]
        if type(value) is dict:
            return {key: javascript(v) for key, v in cast(dict[str, Any], value).items()}
        return value

    session = ExplorerSession(tmp_path)
    first = session.request(
        "construct",
        {
            "controls": controls(session, "dirichlet-concentration-r0"),
            "construction_seed": 7,
            "seed": 11,
            "task_id": 3,
        },
    )
    original = json.loads(first["recipe_json"])["recipe"]
    editor = session.request("mechanism", {})
    draft = javascript(editor["editable_recipe"])
    draft["source"]["network"]["blocks"][-1]["experts"][0]["bias"][0] += 0.125
    valid = session.request("network_draft", {"recipe": draft})
    saved = session.request("edit", {"recipe": javascript(valid["recipe"])})
    after = saved["edit"]["after"]["recipe"]["recipe"]
    assert after["version"] == "prototype-batch-network-realized-v1"
    assert after["population"] == original["population"]
    assert session.request("view", {})["recipe_json"] == first["recipe_json"]


def test_batch_pair_reuses_realized_population_and_curve_uses_observed_inputs(tmp_path: Path):
    import torch

    from r0.modules import priors as p

    session = ExplorerSession(tmp_path)
    first = session.request(
        "construct",
        {
            "controls": controls(session, "dirichlet-concentration-r0"),
            "construction_seed": 7,
            "seed": 11,
            "task_id": 3,
        },
    )
    raw = json.loads(first["recipe_json"])["recipe"]
    paired = session.request("pair", {"intervention": "noise", "value": 0.5})
    after = json.loads(paired["pair"]["after"]["recipe_json"])["recipe"]
    assert after["population"] == raw["population"]
    reopened = ExplorerSession(tmp_path).request("open_pair", {"path": paired["pair"]["path"]})
    assert reopened["pair"]["card"] == paired["pair"]["card"]
    curve = session.request("curve", {"feature": 0, "points": 7})["curve"]
    x = torch.tensor(curve["anchor_inputs"], dtype=torch.float64).repeat(7, 1)
    x[:, 0] = torch.tensor(curve["values"], dtype=torch.float64)
    source = p.recipe_from_dict(raw["request"]["source"])
    assert isinstance(source, p.NetworkRecipe)
    expected = p.evaluate_frozen_function(source.function, x)
    assert torch.equal(expected, torch.tensor(curve["scores"], dtype=torch.float64))
