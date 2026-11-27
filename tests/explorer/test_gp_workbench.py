"""Finite GP populations through the guarded explorer interface."""

import json
from pathlib import Path
from typing import Any

import pytest

from r0.modules.explorer import ExplorerSession


def test_gp_population_construct_replay_and_inspect(tmp_path: Path):
    session = ExplorerSession(tmp_path)
    options = session.request("construction_examples", {})["examples"]
    controls = options["GP columns: linear / regression / shared RBF"]
    first = session.request(
        "construct",
        {
            "controls": controls,
            "construction_seed": "7",
            "seed": "11",
            "task_id": str(2**54 + 3),
        },
    )
    record = json.loads(first["recipe_json"])
    assert record["recipe"]["source"]["version"] == "gp-column-population-v1"
    assert record["task_id"] == 2**54 + 3
    assert first["view"]["construction_details"]["kind"] == "finite-gp-columns"
    replay = session.request("generate", {"record_json": first["recipe_json"]})
    assert replay["recipe_json"] == first["recipe_json"]
    graph = session.request("graph", {"indices": [15, 0, 15]})["graph"]
    assert graph["rows"] == [15, 0, 15]
    assert graph["shared_matrices"][0]["row_ids"] == list(range(16))


def test_gp_replay_and_trace_remain_in_guarded_worker(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    import numpy as np

    session = ExplorerSession(tmp_path)
    controls = session.request("construction_examples", {})["examples"][
        "GP columns: quadratic / regression / per-column RBF"
    ]
    first = session.request(
        "construct",
        {
            "controls": controls,
            "construction_seed": 7,
            "seed": 11,
            "task_id": 3,
        },
    )

    def forbidden(*args: Any, **kwargs: Any):
        raise AssertionError("GP covariance replay escaped the guarded worker")

    monkeypatch.setattr(np.linalg, "cholesky", forbidden)
    session.request("generate", {"record_json": first["recipe_json"]})
    session.request("graph", {"indices": [15, 0, 15]})
    assert session.request("export", {})["data"]


def test_gp_next_explicitly_constructs_new_population_and_keeps_original(tmp_path: Path):
    session = ExplorerSession(tmp_path)
    controls = session.request("construction_examples", {})["examples"][
        "GP columns: neural / multiclass / per-column RBF"
    ]
    first = session.request(
        "construct",
        {
            "controls": controls,
            "construction_seed": 7,
            "seed": "11",
            "task_id": str(2**54 + 3),
        },
    )
    next_task = session.request(
        "generate",
        {
            "record_json": first["recipe_json"],
            "population_identity": {"seed": "11", "task_id": str(2**54 + 4)},
        },
    )
    before = json.loads(first["recipe_json"])["recipe"]["source"]
    after = json.loads(next_task["recipe_json"])["recipe"]["source"]
    assert before["request"] == after["request"]
    assert before["population_id"] != after["population_id"]
    assert before["target"] != after["target"]
    assert after["task_id"] == 2**54 + 4
    replay = session.request("generate", {"record_json": first["recipe_json"]})
    assert replay["recipe_json"] == first["recipe_json"]


def test_gp_invalid_construct_and_timeout_preserve_accepted_population(tmp_path: Path):
    import copy

    session = ExplorerSession(tmp_path)
    controls = session.request("construction_examples", {})["examples"][
        "GP columns: linear / regression / shared RBF"
    ]
    first = session.request(
        "construct",
        {
            "controls": controls,
            "construction_seed": 7,
            "seed": 11,
            "task_id": 3,
        },
    )
    invalid = copy.deepcopy(controls)
    invalid["request"]["rows"] = 17
    with pytest.raises(ValueError, match="Generation failed"):
        session.request(
            "construct",
            {
                "controls": invalid,
                "construction_seed": 7,
                "seed": 11,
                "task_id": 3,
            },
        )
    assert session.request("view", {})["recipe_json"] == first["recipe_json"]
    with pytest.raises(ValueError, match="Generation failed"):
        session.request("generate", {"record_json": first["recipe_json"], "max_seconds": 0.001})
    assert session.request("view", {})["recipe_json"] == first["recipe_json"]
    with pytest.raises(ValueError, match="Generation failed"):
        session.request(
            "generate",
            {
                "record_json": first["recipe_json"],
                "causal_limits": {"working_bytes": 1024**2},
            },
        )
    assert session.request("view", {})["recipe_json"] == first["recipe_json"]
