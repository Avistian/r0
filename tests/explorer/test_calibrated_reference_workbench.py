"""Guarded active and changed-population references retain the original fixed fit."""

import json
from pathlib import Path

import numpy as np
import pytest

from r0.modules.explorer import ExplorerSession


def test_guarded_calibrated_active_and_after_references(tmp_path: Path):
    session = ExplorerSession(tmp_path)
    controls = next(
        v
        for k, v in session.request("construction_examples", {})["examples"].items()
        if k.startswith("Population DAG: functional Conv")
    )
    cap = {"parameters": 32_000_000, "working_bytes": 8 * 1024**3}
    controls["request"]["limits"].update(cap)
    controls["request"]["base"]["limits"].update(cap)
    session.request(
        "construct",
        {
            "controls": controls,
            "construction_seed": "17",
            "seed": "31",
            "task_id": str(2**55 + 41),
            "causal_limits": cap,
        },
    )
    session.request(
        "population_calibrate",
        {
            "calibration_rows": 32,
            "calibration_seed": "71",
            "calibration_task_id": "73",
            "causal_limits": cap,
        },
    )
    before = session.request(
        "population_query",
        {
            "construction_seed": "107",
            "seed": "101",
            "task_id": str(2**55 + 103),
            "engine": "Philox",
            "causal_limits": cap,
        },
    )
    request = {"feature": 0, "anchor_row": 3, "points": 3, "causal_limits": cap, "max_seconds": 120}
    curve = session.request("curve", request)["curve"]
    assert curve["version"] == "calibrated-population-graph-reference-v1"
    assert json.loads(curve["source_json"]) == json.loads(before["recipe_json"])["recipe"]
    assert len(curve["raw_scores"]) == 3
    change = {
        "version": "hard-causal-intervention-v1",
        "node": 1,
        "channels": [0],
        "values": [2.0],
        "category": None,
        "location": "state_and_readout",
    }
    pair = session.request(
        "intervene",
        {
            "interventions_json": json.dumps([change]),
            "mode": "population",
            "population": "11",
            "causal_limits": cap,
            "max_seconds": 120,
        },
    )
    changed = session.request(
        "curve",
        {
            **request,
            "population_reference_side": "after",
            "expected_intervention_id": pair["intervention"]["after_id"],
        },
    )["curve"]
    assert changed["calibration_fingerprint"] == curve["calibration_fingerprint"]
    np.testing.assert_array_equal(changed["raw_scores"], [[6.0]] * 3)
    assert session.intervention is not None
    np.testing.assert_allclose(
        changed["targets"], float(session.intervention.after.targets[3]), atol=1e-13
    )
    assert session.task is not None
    assert session.task.recipe_json == before["recipe_json"]
    with pytest.raises(ValueError, match="identity"):
        session.request("curve", {**request, "expected_episode_id": "wrong"})
    with pytest.raises(ValueError, match="timed out"):
        session.request("curve", {**request, "max_seconds": 0.001})
    assert session.task.recipe_json == before["recipe_json"]
