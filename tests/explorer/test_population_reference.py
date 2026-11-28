"""Population references run only in the worker and preserve saved session state on failure."""

import json
from pathlib import Path
from typing import Any

import pytest

from r0.modules import priors as p
from r0.modules.explorer import ExplorerSession


def test_population_reference_guarded_session_and_comparison(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    session = ExplorerSession(tmp_path)
    examples = session.request("construction_examples", {})["examples"]
    controls = next(
        v for k, v in examples.items() if k.startswith("Population DAG: functional Conv")
    )
    saved = session.request(
        "construct",
        {"controls": controls, "construction_seed": "17", "seed": "31", "task_id": str(2**55 + 7)},
    )
    assert session.task is not None
    original = session.task.recipe_json

    def forbidden(*args: Any, **kwargs: Any):
        pytest.fail("parent must not decode/replay a population reference")

    monkeypatch.setattr(p, "recipe_from_dict", forbidden)
    reply = session.request("curve", {"feature": 1, "anchor_row": 7, "points": 3})["curve"]
    assert reply["task_id"] == str(2**55 + 7)
    assert reply["kind"] == "population_original_row_reference"
    assert len(reply["scores"]) == 3
    assert json.loads(reply["reference_json"])["task_id"] == 2**55 + 7
    assert session.task.recipe_json == original
    with pytest.raises(ValueError, match="stale"):
        session.request("curve", {"feature": 1, "points": 3, "expected_episode_id": "wrong"})
    change = {
        "version": "hard-causal-intervention-v1",
        "node": 2,
        "channels": [0],
        "values": [8.0],
        "category": None,
        "location": "state_and_readout",
    }
    pair = session.request(
        "intervene",
        {"interventions_json": json.dumps([change]), "mode": "population", "population": "3"},
    )
    expected = pair["intervention"]["after_id"]
    request = {
        "feature": 1,
        "anchor_row": 7,
        "points": 3,
        "population_reference_side": "after",
        "expected_intervention_id": expected,
    }
    after = session.request("curve", request)["curve"]
    assert after["scores"] == [[8.0]] * 3
    assert after["population_mode"] == "population"
    assert after["population_index"] == "3"
    with pytest.raises(ValueError, match="stale"):
        session.request("curve", {**request, "expected_intervention_id": "wrong"})
    with pytest.raises(ValueError, match="reference|limit|metadata"):
        session.request("curve", {**request, "capture_location": "implicit"})
    with pytest.raises(ValueError, match="timed out"):
        session.request("curve", {**request, "max_seconds": 0.01})
    assert session.task.recipe_json == original
    assert session.request("view", {})["intervention"]["after_id"] == expected
    assert saved["view"]["episode_id"] == session.task.episode_id
    assert session.request("intervention_export", {"expected_intervention_id": expected})["data"]
