"""Known GP-root producer controls reject before covariance reconstruction."""

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from r0.modules import generation as g
from r0.modules import priors as p
from r0.modules.explorer import ExplorerSession


def fixture(tmp_path: Path):
    examples = ExplorerSession(tmp_path).request("construction_examples", {})["examples"]
    controls = next(
        v
        for k, v in examples.items()
        if k.startswith("GP root DAG: functional linear / regression")
    )
    graph = g.GraphConstruction(json.dumps(controls), 17).build(
        p.CausalLimits(), episode_seed=31, task_id=2**55 + 7
    )
    task = p.generate_task(graph, seed=31, task_id=2**55 + 7)
    return controls, json.loads(task.recipe_json)


@pytest.mark.parametrize(
    "case",
    [
        "seed",
        "float_id",
        "nested_id",
        "request",
        "binding",
        "matrix",
        "task",
        "execution",
        "version",
        "schema",
        "source_schema",
        "caps",
        "root",
        "scope",
    ],
)
def test_gp_root_known_producer_mismatch_before_cholesky(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, case: str
):
    controls, record = fixture(tmp_path)
    raw = record["recipe"]
    if case == "seed":
        record["seed"] = True
    elif case == "float_id":
        record["task_id"] = float(2**55 + 7)
    elif case == "nested_id":
        raw["source"]["task_id"] += 1
    elif case == "request":
        raw["source"]["request"]["target_family"] = "quadratic"
    elif case == "binding":
        raw["bindings"][0]["columns"] = [1, 0]
    elif case == "matrix":
        raw["base"]["nodes"][0]["function"]["matrix"][0][0] = 9.0
    elif case == "task":
        raw["base"]["task"]["noise_ratio"] = 0.73
    elif case == "execution":
        record["execution"]["implementation"] = "wrong"
    elif case == "version":
        raw["version"] = "wrong"
    elif case == "schema":
        record["unexpected"] = 1
    elif case == "source_schema":
        raw["source"]["unexpected"] = 1
    elif case == "caps":
        raw["limits"]["working_bytes"] -= 1
    elif case == "root":
        raw["base"]["nodes"][0]["root"]["family"] = "uniform"
    else:
        raw["base"]["nodes"][1]["parents"] = []

    def forbidden(*args: Any, **kwargs: Any):
        pytest.fail("known producer mismatch began covariance work")

    monkeypatch.setattr(np.linalg, "cholesky", forbidden)
    assert not p.gp_root_construction_matches(
        controls, record, seed=31, task_id=2**55 + 7, limits=p.CausalLimits()
    )


def test_gp_root_matching_keeps_valid_and_omitted_defaults(tmp_path: Path):
    controls, record = fixture(tmp_path)
    original = json.dumps(record, sort_keys=True)
    assert p.gp_root_construction_matches(
        controls, record, seed=31, task_id=2**55 + 7, limits=p.CausalLimits()
    )
    del controls["base"]["task"]["noise_ratio"]
    assert p.gp_root_construction_matches(
        controls, record, seed=31, task_id=2**55 + 7, limits=p.CausalLimits()
    )
    record["recipe"]["base"]["task"]["noise_ratio"] = 0.73
    assert not p.gp_root_construction_matches(
        controls, record, seed=31, task_id=2**55 + 7, limits=p.CausalLimits()
    )
    assert json.loads(original)["recipe"]["limits"] == asdict(p.CausalLimits())
