"""Sampled native construction stays typed through the public workbench lifecycle."""

import base64
import csv
import io
import json
import zipfile
from pathlib import Path

import pytest

from r0.modules.explorer import ExplorerSession


@pytest.mark.parametrize("law", ["uniform", "dirichlet", "logistic-normal"])
def test_sampled_native_construct_export_edit_and_reopen(tmp_path: Path, law: str) -> None:
    session = ExplorerSession(tmp_path)
    controls = session.request("construction_examples", {})["examples"][
        "Native mixed graph (sampled domains)"
    ]
    controls.update(rows=16, context_rows=8, calibration_rows=16)
    for domain in controls["domains"]:
        domain["frequency_laws"] = [law]
    result = session.request(
        "construct", {"controls": controls, "construction_seed": 28, "seed": 19, "task_id": 7}
    )
    view = result["view"]
    assert view["target_origin"] == "native-variable-v1"
    assert [f["kind"] for f in view["feature_schema"]] == ["continuous", "binary", "categorical"]
    recipe = json.loads(result["recipe_json"])["recipe"]
    assert recipe["version"] == "sampled-native-graph-v1"
    assert recipe["request"]["domains"][2]["frequency_laws"] == [law]
    assert len(recipe["initial"]) == len(recipe["final"]) == 4
    assert view["native_construction"] == "sampled-native-graph-v1"
    assert (
        view["target_frequency_weights"] == recipe["recipe"]["nodes"][-1]["domain"]["frequencies"]
    )
    assert session.request("graph", {"indices": [7, 0, 7]})["graph"]["rows"] == [7, 0, 7]
    row = session.request("row", {"index": 0})["row"]
    assert row["display_inputs"][2] in view["feature_schema"][2]["vocabulary"]
    with zipfile.ZipFile(io.BytesIO(base64.b64decode(session.request("export", {})["data"]))) as z:
        identity = json.loads(z.read("identity.json"))
        assert identity["target_origin"] == "native-variable-v1"
        rows = list(csv.DictReader(io.StringIO(z.read("predictor.csv").decode())))
        assert all(r["context_target"] == "" for r in rows if r["role"] == "query")
        assert all(r["x2"] in view["feature_schema"][2]["vocabulary"] for r in rows)
    reopened = ExplorerSession(tmp_path).request(
        "open", {"archive": result["archive"], "task_id": 7}
    )
    assert reopened["recipe_json"] == result["recipe_json"]
    editor = session.request("mechanism", {})
    assert editor["construction"] == recipe["version"]
    assert "frequency" in editor["notice"].lower()
    edited = editor["editable_recipe"]
    edited["nodes"][-1]["domain"].update(emission="argmax", temperature=1.0)
    compared = session.request("edit", {"recipe": edited})
    assert compared["view"]["episode_id"] == view["episode_id"]
    assert compared["edit"]["card"]["after_episode_id"] != view["episode_id"]
    assert compared["edit"]["after"]["target_origin"] == "native-variable-v1"
    assert session.request("edit_export", {})["data"]
    again = ExplorerSession(tmp_path).request("open_edit", {"path": compared["edit"]["path"]})
    assert again["edit"]["card"] == compared["edit"]["card"]
    controls["domains"][-1]["cardinalities"] = [101]
    with pytest.raises(ValueError, match="Generation failed"):
        session.request(
            "construct", {"controls": controls, "construction_seed": 28, "seed": 19, "task_id": 7}
        )
    assert session.request("view", {})["view"]["episode_id"] == view["episode_id"]


@pytest.mark.parametrize("kind", ["continuous", "binary"])
def test_sampled_native_derives_target_type(tmp_path: Path, kind: str) -> None:
    session = ExplorerSession(tmp_path)
    controls = session.request("construction_examples", {})["examples"][
        "Native mixed graph (sampled domains)"
    ]
    controls.update(rows=16, context_rows=8, calibration_rows=16)
    controls["domains"][-1]["kinds"] = [kind]
    result = session.request(
        "construct", {"controls": controls, "construction_seed": 29, "seed": 19, "task_id": 7}
    )
    assert result["view"]["task_type"] == ("regression" if kind == "continuous" else "binary")
    assert len(result["view"]["classes"]) == (0 if kind == "continuous" else 2)
    assert result["view"]["target_origin"] == "native-variable-v1"
