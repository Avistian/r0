"""Local and callback typed-converter workflows preserve original frozen archives."""

import json
from pathlib import Path

import pytest

from r0.modules.explorer import ExplorerSession


@pytest.mark.parametrize("mode", ["neighbor_disc", "softmax_int"])
def test_converted_construction_save_reopen_inspect_and_authored_edit(
    tmp_path: Path, mode: str
) -> None:
    session = ExplorerSession(tmp_path)
    controls = session.request("construction_examples", {})["examples"][
        "Typed categorical converter pipelines"
    ]
    controls["request"].update(rows=16, context_rows=8, calibration_rows=16)
    controls["request"]["converter_prior"]["modes"] = [mode]
    result = session.request(
        "construct", {"controls": controls, "construction_seed": 28, "seed": 19, "task_id": 7}
    )
    assert result["view"]["converted"]
    assert result["view"]["target_emission"] in ("nearest prototype", "softmax")
    assert session.request("graph", {"indices": [15, 0, 15]})["graph"]["rows"] == [15, 0, 15]
    reopened = ExplorerSession(tmp_path).request(
        "open", {"archive": result["archive"], "task_id": 7}
    )
    assert reopened["recipe_json"] == result["recipe_json"]
    editor = session.request("mechanism", {})
    assert editor["converted"]
    editor["editable_recipe"]["nodes"][0]["equation"]["pipeline"]["node_importance"] *= 2
    edited = session.request("edit", {"recipe": editor["editable_recipe"]})
    assert edited["recipe_json"] == result["recipe_json"]
    assert edited["edit"]["card"]["kind"] == "mechanism-edit"
    assert json.loads(result["recipe_json"])["recipe"]["version"] == "sampled-converted-graph-v1"

    info = session.request("intervention_controls", {})
    assert info["converted"]
    target = json.loads(result["recipe_json"])["recipe"]["recipe"]["target"]
    node = info["nodes"][target["node"]]
    converter = next(c for c in node["converters"] if c["name"] == target["capture"])
    changes = [
        {
            "version": "converter-hard-intervention-v1",
            "node": target["node"],
            "capture": target["capture"],
            "category": converter["vocabulary"][1],
            "location": "capture",
        }
    ]
    pair = session.request("intervene", {"interventions_json": json.dumps(changes)})
    assert pair["recipe_json"] == result["recipe_json"]
    assert pair["intervention"]["after"]["target_emission"].startswith("forced category")
    assert all(r["target"] == 1 for r in pair["intervention"]["after"]["rows"])
    trace = session.request(
        "intervention_graph",
        {"indices": [15, 0, 15], "expected_intervention_id": pair["intervention"]["after_id"]},
    )
    assert trace["after"]["rows"] == [15, 0, 15]
    saved = ExplorerSession(tmp_path).request(
        "open_intervention", {"path": pair["intervention"]["path"]}
    )
    assert saved["intervention"]["card"] == pair["intervention"]["card"]
    with pytest.raises(ValueError, match="previous dataset/comparison retained"):
        session.request("intervene", {"interventions_json": "[]"})
    assert (
        session.request("view", {})["intervention"]["after_id"] == pair["intervention"]["after_id"]
    )
