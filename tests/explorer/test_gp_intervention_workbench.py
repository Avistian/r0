"""GP comparisons stay in workers and retain the active source on failure."""

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from r0.modules import priors as p
from r0.modules.explorer import ExplorerSession


@pytest.mark.parametrize("fitted", [False, True])
@pytest.mark.parametrize("family", ["functional", "native", "vector", "converted"])
def test_gp_intervention_guarded_pair_and_reopen(
    tmp_path: Path, family: str, fitted: bool, monkeypatch: pytest.MonkeyPatch
):
    session = ExplorerSession(tmp_path)
    catalog = session.request("construction_examples", {})["examples"]
    controls = next(
        v for k, v in catalog.items() if k.startswith(f"GP root DAG: {family} linear / regression")
    )
    baseline = session.request(
        "construct",
        {"controls": controls, "construction_seed": 0, "seed": "31", "task_id": str(2**55 + 7)},
    )
    if fitted:
        session.request(
            "gp_root_calibrate",
            {"calibration_rows": 37, "calibration_seed": "41", "calibration_task_id": "43"},
        )
        baseline = session.request(
            "gp_root_query", {"seed": "31", "task_id": str(2**55 + 9), "locations": None}
        )
    change = (
        p.NativeVectorHardIntervention(0, 0, value=2.5)
        if family == "vector"
        else p.HardIntervention(
            0, (0,), (2.5,), location="state" if family == "native" else "state_and_readout"
        )
    )
    from dataclasses import asdict

    def forbidden(*args: Any, **kwargs: Any):
        raise AssertionError("GP intervention escaped its worker")

    monkeypatch.setattr(np.linalg, "cholesky", forbidden)
    result = session.request(
        "intervene", {"interventions_json": json.dumps([asdict(change)]), "mode": "matched"}
    )
    assert result["recipe_json"] == baseline["recipe_json"]
    assert session.intervention is not None
    after = session.intervention.after
    before = session.intervention.before
    np.testing.assert_array_equal(after.inputs[:, 1], before.inputs[:, 1])
    expected = 5 - before.inputs[:, 1]
    if fitted:
        calibration = json.loads(before.recipe_json)["recipe"]["source"]["calibration"]
        expected = (expected - calibration["center"][0]) / calibration["scale"][0]
    np.testing.assert_allclose(after.targets, expected)
    pair = result["intervention"]
    trace = session.request(
        "intervention_graph", {"indices": [13, 2, 13], "expected_intervention_id": pair["after_id"]}
    )
    assert trace["after"]["rows"] == [13, 2, 13]
    with pytest.raises(ValueError, match="Intervention create failed"):
        session.request("intervene", {"interventions_json": "[{}]", "mode": "matched"})
    assert session.request("view", {})["intervention"]["path"] == pair["path"]
    assert session.request("intervention_export", {"expected_intervention_id": pair["after_id"]})[
        "data"
    ]
    reopened = session.request("open_intervention", {"path": pair["path"]})
    assert reopened["recipe_json"] == baseline["recipe_json"]

    after_archive = session.intervention.document["after_archive"]
    opened = session.request(
        "open", {"archive": after_archive, "task_id": str(2**55 + (9 if fitted else 7))}
    )
    assert opened["view"]["recipe"]["recipe"]["version"] == "gp-root-intervention-v1"
    assert session.request("graph", {"indices": [13, 2, 13]})["graph"]["rows"] == [13, 2, 13]
    replay = session.request("gp_root_replay", {})
    assert replay["recipe_json"] == opened["recipe_json"]
    if fitted:
        fit_trace = session.request("graph", {"indices": [36, 1, 36], "population": "fit"})["graph"]
        assert fit_trace["rows"] == [36, 1, 36]
        assert str(fit_trace["fit_seed"]) == "41"
        assert fit_trace["query_fit"] is False
