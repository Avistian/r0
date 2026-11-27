"""GP target queries stay in the guarded worker without changing population archives."""

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from r0.modules.explorer import ExplorerSession


def test_gp_target_curve_guard_identity_export_and_failure_preservation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    session = ExplorerSession(tmp_path)
    controls = session.request("construction_examples", {})["examples"][
        "GP columns: neural / binary / per-column RBF"
    ]
    baseline = session.request(
        "construct",
        {
            "controls": controls,
            "construction_seed": 7,
            "seed": "18014398509482001",
            "task_id": "18014398509482003",
        },
    )
    record = json.loads(baseline["recipe_json"])

    def forbidden(*args: Any, **kwargs: Any):
        raise AssertionError("GP target reference replay escaped guarded worker")

    monkeypatch.setattr(np.linalg, "cholesky", forbidden)
    curve = session.request(
        "curve", {"feature": 1, "anchor_row": 3, "points": 9, "bounds": [-1.0, 2.0]}
    )["curve"]
    assert curve["kind"] == "gp_column_target_reference"
    assert curve["task_id"] == "18014398509482003"
    assert curve["population_id"] == record["recipe"]["source"]["population_id"]
    assert curve["feature_role"] == "joint GP column"
    assert len(curve["probabilities"][0]) == 2
    for invalid in (
        {"points": 1},
        {"bounds": [2.0, -2.0]},
        {"max_seconds": 0.001},
        {"causal_limits": {"working_bytes": 1024**2}},
    ):
        with pytest.raises(ValueError, match="points|bounds|timed out|failed|limit"):
            session.request("curve", {"feature": 1, "anchor_row": 3, "points": 512, **invalid})
        assert session.request("view", {})["recipe_json"] == baseline["recipe_json"]
    assert session.request("export", {})["data"]
    reopened = session.request(
        "open", {"archive": baseline["archive"], "task_id": "18014398509482003"}
    )
    assert reopened["recipe_json"] == baseline["recipe_json"]
