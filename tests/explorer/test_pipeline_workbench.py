"""Optional numerical pipelines survive the public workbench lifecycle."""

import json
from pathlib import Path

import pytest

from r0.modules.explorer import ExplorerSession


@pytest.mark.parametrize("variant", ["historical-tuple", "historical-graph", "readout-only"])
def test_sampled_pipeline_construct_edit_and_reopen(tmp_path: Path, variant: str) -> None:
    session = ExplorerSession(tmp_path)
    controls = session.request("construction_examples", {})["examples"][
        f"Numerical pipeline: {variant}"
    ]
    controls["task"].update(rows=16, context_rows=8)
    controls["calibration_rows"] = 4
    controls["width_prior"].update(low=1, high=3)
    controls["pipeline_prior"]["gaussian_noise"] = True
    result = session.request(
        "construct", {"controls": controls, "construction_seed": 28, "seed": 19, "task_id": 0}
    )
    recipe = json.loads(result["recipe_json"])["recipe"]
    assert recipe["request"]["version"] == "geometry-pipeline-request-v1"
    assert recipe["request"]["pipeline_prior"]["converter"]["variant"] == variant
    assert recipe["pipelines"]
    reopened = ExplorerSession(tmp_path).request(
        "open", {"archive": result["archive"], "task_id": 0}
    )
    assert reopened["recipe_json"] == result["recipe_json"]
    assert session.request("graph", {"indices": [7, 0, 7]})["graph"]["rows"] == [7, 0, 7]
    editor = session.request("mechanism", {})
    editor["editable_recipe"]["nodes"][0]["pipeline"]["node_importance"] = 2.0
    changed = session.request("edit", {"recipe": editor["editable_recipe"]})
    assert changed["edit"]["card"]["after_episode_id"] != result["view"]["episode_id"]
    assert changed["view"]["episode_id"] == result["view"]["episode_id"]
    assert session.request("export", {})["data"]
    assert session.request("edit_export", {})["data"]
    again = ExplorerSession(tmp_path).request("open_edit", {"path": changed["edit"]["path"]})
    assert again["edit"]["card"] == changed["edit"]["card"]
