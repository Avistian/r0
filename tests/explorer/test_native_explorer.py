"""Native variables remain typed through public workbench and export actions."""

import base64
import csv
import io
import json
import zipfile
from copy import deepcopy
from pathlib import Path

import pytest

from r0.modules.explorer import ExplorerSession


def test_native_construct_inspect_export_edit_and_reopen(tmp_path: Path) -> None:
    session = ExplorerSession(tmp_path)
    controls = session.request("construction_examples", {})["examples"][
        "Native mixed graph (authored)"
    ]
    controls["task"].update(rows=16, context_rows=8)
    result = session.request(
        "construct", {"controls": controls, "construction_seed": 28, "seed": 19, "task_id": 7}
    )
    view = result["view"]
    assert view["target_origin"] == "native-variable-v1"
    assert [f["kind"] for f in view["feature_schema"]] == ["continuous", "binary", "categorical"]
    assert view["features"][2]["summary"]["categories"] == ["red", "green", "blue"]
    row = session.request("row", {"index": 0})["row"]
    assert row["display_inputs"][2] in ("red", "green", "blue")
    assert type(row["display_inputs"][1]) is int
    assert row["target_label"] in view["classes"]
    graph = session.request("graph", {"indices": [7, 0, 7]})["graph"]
    assert graph["rows"] == [7, 0, 7]
    assert graph["target_origin"] == "native-variable-v1"
    exported = session.request("export", {})
    with zipfile.ZipFile(io.BytesIO(base64.b64decode(exported["data"]))) as bundle:
        identity = json.loads(bundle.read("identity.json"))
        assert identity["feature_schema"] == json.loads(json.dumps(view["feature_schema"]))
        assert identity["target_origin"] == view["target_origin"]
        rows = list(csv.DictReader(io.StringIO(bundle.read("predictor.csv").decode())))
        assert all(r["x2"] in ("red", "green", "blue") for r in rows)
        assert all(r["x1"] in ("0", "1") for r in rows)
        assert all(r["context_target"] == "" for r in rows if r["role"] == "query")
        assert all(r["context_target"] in view["classes"] for r in rows if r["role"] == "context")
    reopened = ExplorerSession(tmp_path).request(
        "open", {"archive": result["archive"], "task_id": 7}
    )
    assert reopened["view"]["feature_schema"] == view["feature_schema"]
    assert reopened["recipe_json"] == result["recipe_json"]
    editor = session.request("mechanism", {})
    edited = editor["editable_recipe"]
    edited["nodes"][-1]["domain"]["emission"] = "argmax"
    compared = session.request("edit", {"recipe": edited})
    assert compared["edit"]["card"]["kind"] == "mechanism-edit"
    assert compared["edit"]["after"]["target_origin"] == "native-variable-v1"
    assert compared["view"]["episode_id"] == view["episode_id"]
    again = ExplorerSession(tmp_path).request("open_edit", {"path": compared["edit"]["path"]})
    assert again["edit"]["card"] == compared["edit"]["card"]
    with pytest.raises(ValueError, match="native graph pairs"):
        session.request("pair", {"intervention": "separation", "value": 2})


@pytest.mark.parametrize(("target", "kind", "classes"), [(0, "regression", 0), (1, "binary", 2)])
def test_native_direct_continuous_and_binary_targets(
    tmp_path: Path, target: int, kind: str, classes: int
) -> None:
    session = ExplorerSession(tmp_path)
    controls = session.request("construction_examples", {})["examples"][
        "Native mixed graph (authored)"
    ]
    controls["task"].update(rows=16, context_rows=8, task_type=kind, classes=classes)
    nodes = list(controls["nodes"])
    nodes[3] = deepcopy(nodes[target])
    nodes[3]["equation"]["parents"] = (0,)
    nodes[3]["equation"]["root"] = None
    controls["nodes"] = nodes
    result = session.request(
        "construct", {"controls": controls, "construction_seed": 28, "seed": 19, "task_id": 0}
    )
    view = result["view"]
    for row in view["rows"]:
        assert row["target"] == row["inputs"][target]
        if classes:
            assert row["target_label"] == ("no", "yes")[int(row["target"])]
        else:
            assert row["target_label"] == row["target"]
    assert view["target_origin"] == "native-variable-v1"
