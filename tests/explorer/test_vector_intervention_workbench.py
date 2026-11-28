"""Vector intervention controls and effective target metadata through guarded sessions."""

import json
from pathlib import Path

import pytest

from r0.modules.explorer import ExplorerSession


def test_vector_intervention_session_domains_siblings_and_failed_receipt(tmp_path: Path) -> None:
    session = ExplorerSession(tmp_path)
    controls = session.request("construction_examples", {})["examples"][
        "Native vector parent composition"
    ]
    controls.update(rows=16, context_rows=8, calibration_rows=8)
    baseline = session.request(
        "construct", {"controls": controls, "construction_seed": 28, "seed": 19, "task_id": 7}
    )
    info = session.request("intervention_controls", {})
    assert info["vector"]
    assert len(info["nodes"][1]["domains"]) == 3
    # Clamp sibling continuous variable, not the discrete target in the same node.
    sibling = {
        "version": "native-vector-hard-intervention-v1",
        "node": 1,
        "variable": 1,
        "value": 4.0,
        "category": None,
    }
    result = session.request(
        "intervene", {"interventions_json": json.dumps([sibling]), "max_seconds": 30}
    )
    assert result["intervention"]["after"]["target_emission"] == "softmax"
    assert all(row["inputs"][3] == 4 for row in result["intervention"]["after"]["rows"])
    domain = dict(info["nodes"][1]["domains"][0], emission="argmax", temperature=1.0)
    change = {
        "version": "native-vector-domain-intervention-v1",
        "node": 1,
        "variable": 0,
        "domain": domain,
    }
    result = session.request(
        "intervene", {"interventions_json": json.dumps([change]), "max_seconds": 30}
    )
    after = result["intervention"]["after"]
    assert after["target_emission"] == "argmax"
    assert after["original_target_domain"]["emission"] == "softmax"
    assert after["effective_target_domain"]["emission"] == "argmax"
    assert result["recipe_json"] == baseline["recipe_json"]
    pair = result["intervention"]
    assert (
        session.request("open_intervention", {"path": pair["path"], "max_seconds": 30})[
            "intervention"
        ]["after_id"]
        == pair["after_id"]
    )
    assert session.request(
        "intervention_export", {"expected_intervention_id": pair["after_id"], "max_seconds": 30}
    )["data"]
    with pytest.raises(ValueError, match="Intervention"):
        session.request(
            "intervene",
            {"interventions_json": json.dumps([dict(sibling, variable=100)]), "max_seconds": 30},
        )
    assert session.request("view", {})["view"]["episode_id"] == baseline["view"]["episode_id"]
    assert list((tmp_path / ".intervention-attempts").glob("*-failed.json"))
