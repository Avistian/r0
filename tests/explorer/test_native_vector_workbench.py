"""Heterogeneous vector JSON controls preserve typed saved data and authored edits."""

import base64
import csv
import io
import json
import zipfile
from pathlib import Path

import pytest

from r0.modules.explorer import ExplorerSession


@pytest.mark.parametrize("parent", [False, True])
def test_vector_workbench_construct_inspect_edit_export_reopen(
    parent: bool, tmp_path: Path
) -> None:
    session = ExplorerSession(tmp_path)
    label = "Native vector parent composition" if parent else "Native heterogeneous vector graph"
    controls = session.request("construction_examples", {})["examples"][label]
    controls.update(rows=16, context_rows=8, calibration_rows=8)
    result = session.request(
        "construct", {"controls": controls, "construction_seed": 28, "seed": 19, "task_id": 7}
    )
    view = result["view"]
    assert [f["kind"] for f in view["feature_schema"]] == [
        "continuous",
        "binary",
        "categorical",
        "continuous",
    ]
    assert len(view["classes"]) == 5
    assert view["construction_details"]["kind"] == "native-vector"
    graph = session.request("graph", {"indices": [7, 0, 7]})["graph"]
    assert graph["rows"] == [7, 0, 7]
    assert next(n for n in graph["nodes"] if n["id"] == "state-0")["shape"] == [3, 6]
    assert (
        ExplorerSession(tmp_path).request("open", {"archive": result["archive"], "task_id": 7})[
            "recipe_json"
        ]
        == result["recipe_json"]
    )
    with zipfile.ZipFile(io.BytesIO(base64.b64decode(session.request("export", {})["data"]))) as z:
        rows = list(csv.DictReader(io.StringIO(z.read("predictor.csv").decode())))
        assert all(r["context_target"] == "" for r in rows if r["role"] == "query")
        assert all(r["x2"] in view["feature_schema"][2]["vocabulary"] for r in rows)
    editor = session.request("mechanism", {})
    assert editor["editable_recipe"]["version"] == "native-vector-causal-v1"
    edited = editor["editable_recipe"]
    edited["nodes"][1]["domains"][0].update(emission="argmax", temperature=1.0)
    comparison = session.request("edit", {"recipe": edited})
    assert comparison["edit"]["after"]["target_origin"] == "native-variable-v1"
    assert session.request("edit_export", {})["data"]
    again = ExplorerSession(tmp_path).request("open_edit", {"path": comparison["edit"]["path"]})
    assert again["edit"]["card"] == comparison["edit"]["card"]
    original = json.loads(result["recipe_json"])
    assert original["recipe"]["recipe"]["nodes"][1]["domains"][0]["emission"] == "softmax"
    controls["target"] = [1, 1]
    with pytest.raises(ValueError, match="Generation failed"):
        session.request(
            "construct", {"controls": controls, "construction_seed": 28, "seed": 19, "task_id": 7}
        )
    assert session.request("view", {})["view"]["episode_id"] == view["episode_id"]
