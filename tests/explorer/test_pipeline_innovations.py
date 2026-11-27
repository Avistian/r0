"""Shared workflow exposes complete innovation laws and preserves failed attempts."""

import json
from pathlib import Path

import pytest

from r0.modules.explorer import ExplorerSession


@pytest.mark.parametrize("family", ["uniform", "student_t", "mixture", "ball"])
def test_innovation_presets_save_inspect_reopen_and_edit(tmp_path: Path, family: str) -> None:
    session = ExplorerSession(tmp_path)
    controls = session.request("construction_examples", {})["examples"][
        f"Pipeline innovations: {family} (r0 extension)"
    ]
    controls["task"].update(rows=16, context_rows=8)
    controls["calibration_rows"] = 8
    controls["width_prior"].update(low=1, high=3)
    controls["pipeline_prior"]["innovation"].update(
        truncation=[None], degrees=[3.0], powers=[1.0], strengths=[0.1], factors=["identity"]
    )
    result = session.request(
        "construct", {"controls": controls, "construction_seed": 28, "seed": 19, "task_id": 7}
    )
    assert result["view"]["construction_details"]["kind"] == "r0-innovation-pipelines"
    assert "r0 innovation" in result["view"]["construction_details"]["notice"]
    graph = session.request("graph", {"indices": [15, 0, 15]})["graph"]
    assert graph["rows"] == [15, 0, 15]
    reopened = ExplorerSession(tmp_path).request(
        "open", {"archive": result["archive"], "task_id": 7}
    )
    assert reopened["recipe_json"] == result["recipe_json"]
    original = json.loads(result["recipe_json"])
    assert original["recipe"]["pipelines"][0]["plan"]["innovation"]["family"] == family
    editor = session.request("mechanism", {})
    editor["editable_recipe"]["nodes"][0]["noise_scale"] = [0.0] * editor["nodes"][0]["width"]
    edited = session.request("edit", {"recipe": editor["editable_recipe"]})
    assert edited["recipe_json"] == result["recipe_json"]
    assert edited["edit"]["card"]["kind"] == "mechanism-edit"
    controls["pipeline_prior"]["innovation"]["degrees"] = [0.0]
    with pytest.raises(ValueError, match="Generation failed"):
        session.request(
            "construct", {"controls": controls, "construction_seed": 28, "seed": 19, "task_id": 7}
        )
    assert session.request("view", {})["view"]["episode_id"] == result["view"]["episode_id"]
