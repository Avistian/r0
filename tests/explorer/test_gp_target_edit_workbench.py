"""GP target edits and all replay remain guarded while the original stays active."""

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from r0.modules.explorer import ExplorerSession


def test_gp_target_editor_guard_reopen_export_and_rollback(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    session = ExplorerSession(tmp_path)
    controls = session.request("construction_examples", {})["examples"][
        "GP columns: quadratic / multiclass / shared RBF"
    ]
    baseline = session.request(
        "construct",
        {
            "controls": controls,
            "construction_seed": 7,
            "seed": str(2**54 + 1),
            "task_id": str(2**54 + 3),
        },
    )
    target = json.loads(baseline["recipe_json"])["recipe"]["source"]["target"]
    target["weights"][0][0] += 1.3

    def forbidden(*args: Any, **kwargs: Any):
        raise AssertionError("GP edit replay escaped worker")

    monkeypatch.setattr(np.linalg, "cholesky", forbidden)
    result = session.request(
        "gp_target_edit",
        {"target_json": json.dumps(target), "mode": "fresh", "fresh_seed": str(2**54 + 7)},
    )
    pair = result["intervention"]
    assert result["recipe_json"] == baseline["recipe_json"]
    assert pair["card"]["fresh_seed"] == str(2**54 + 7)
    assert pair["before"]["rows"] != baseline["view"]["rows"]
    assert [r["inputs"] for r in pair["before"]["rows"]] == [
        r["inputs"] for r in pair["after"]["rows"]
    ]
    for invalid in (
        {"target_json": "{}"},
        {"fresh_seed": True},
        {"mode": "unknown"},
        {"max_seconds": 0.001},
    ):
        with pytest.raises(ValueError, match="GP|archive|limit|geometry|Intervention"):
            session.request(
                "gp_target_edit",
                {"target_json": json.dumps(target), "mode": "fresh", "fresh_seed": "7", **invalid},
            )
        assert session.request("view", {})["recipe_json"] == baseline["recipe_json"]
        assert session.request("view", {})["intervention"]["path"] == pair["path"]
    trace = session.request(
        "intervention_graph", {"indices": [7, 2, 7], "expected_intervention_id": pair["after_id"]}
    )
    assert trace["after"]["rows"] == [7, 2, 7]
    assert session.request("intervention_export", {"expected_intervention_id": pair["after_id"]})[
        "data"
    ]
    reopened = session.request("open_intervention", {"path": pair["path"]})
    assert reopened["view"]["recipe"]["recipe"]["version"] == "gp-column-target-recipe-v1"
    replay = session.request("generate", {"record_json": reopened["recipe_json"]})
    assert replay["recipe_json"] == reopened["recipe_json"]
    assert session.request("graph", {"indices": [3, 1, 3]})["graph"]["rows"] == [3, 1, 3]
