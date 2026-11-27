"""Explicit asymmetric JSON controls survive guarded construction and saved reopen."""

import json
from pathlib import Path

import pytest

from r0.modules.explorer import ExplorerSession


def test_asymmetric_controls_reopen_inspect_export_and_failure_receipt(tmp_path: Path) -> None:
    session = ExplorerSession(tmp_path)
    controls = session.request("construction_examples", {})["examples"][
        "Parent composition: native domains"
    ]
    controls.update(rows=16, context_rows=8, calibration_rows=16)
    for domain in controls["domains"]:
        if tuple(domain["kinds"]) == ("continuous",):
            continue
        counts = set(domain["cardinalities"]) if "categorical" in domain["kinds"] else {2}
        domain.update(
            version="native-asymmetric-domain-prior-v1",
            frequency_laws=["asymmetric-dirichlet"],
            concentration_vectors=[[float(i + 1) for i in range(k)] for k in counts],
        )
    result = session.request(
        "construct", {"controls": controls, "construction_seed": 28, "seed": 19, "task_id": 7}
    )
    assert result["view"]["target_origin"] == "native-variable-v1"
    saved = json.loads(result["recipe_json"])["recipe"]
    assert saved["request"]["domains"][-1]["version"] == "native-asymmetric-domain-prior-v1"
    assert session.request("graph", {"indices": [3, 1, 3]})["graph"]["rows"] == [3, 1, 3]
    assert session.request("export", {})["data"]
    assert (
        ExplorerSession(tmp_path).request("open", {"archive": result["archive"], "task_id": 7})[
            "recipe_json"
        ]
        == result["recipe_json"]
    )
    controls["domains"][-1]["concentration_vectors"][0][0] = 0
    with pytest.raises(ValueError, match="Generation failed"):
        session.request(
            "construct", {"controls": controls, "construction_seed": 28, "seed": 19, "task_id": 7}
        )
    assert session.request("view", {})["view"]["episode_id"] == result["view"]["episode_id"]
