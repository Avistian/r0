"""Guarded fresh query construction keeps the original independent fit."""

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from test_gp_calibration_construction import controls

from r0.modules import generation as g
from r0.modules import priors as p


@pytest.mark.parametrize("family", ["functional", "native", "vector", "converted"])
def test_gp_query_constructor_literal(tmp_path: Path, family: str):
    fit = g.GraphConstruction(json.dumps(controls(tmp_path, family)), 0).build(
        p.CausalLimits(), episode_seed=31, task_id=2**55 + 7
    )
    draft = dict(topology="query-calibrated-gp-root", source=asdict(fit), locations=None)
    construction = g.GraphConstruction(json.dumps(draft), 0)
    assert construction.task_record()["rows"] == 16
    query = construction.build(p.CausalLimits(), episode_seed=47, task_id=2**55 + 11)
    assert isinstance(query, p.GPRootCalibrationQuery)
    assert asdict(query.source) == asdict(fit)
    raw = np.asarray(query.original.source.draw.values)
    fit_raw = np.asarray(query.fit.source.draw.values)
    reference = 2 * fit_raw[:, 0] - fit_raw[:, 1]
    task = p.generate_task(query, seed=47, task_id=2**55 + 11)
    np.testing.assert_allclose(
        task.targets, (2 * raw[:, 0] - raw[:, 1] - reference.mean()) / reference.std(), atol=1e-12
    )


def test_guarded_query_keeps_fit_replay_and_failure_state(tmp_path: Path):
    from r0.modules.explorer import ExplorerSession

    session = ExplorerSession(tmp_path)
    fit = g.GraphConstruction(json.dumps(controls(tmp_path / "catalog")), 0).build(
        p.CausalLimits(), episode_seed=31, task_id=2**55 + 7
    )
    session.request(
        "generate", dict(recipe=asdict(fit), seed="31", task_id=str(2**55 + 7), max_seconds=120)
    )
    result = session.request(
        "gp_root_query", dict(seed="47", task_id=str(2**55 + 11), locations=None, max_seconds=120)
    )
    query = result["view"]["recipe"]["recipe"]
    assert query["version"] == "gp-root-calibration-query-v1"
    assert session.task is not None
    original = session.task
    replay = session.request("gp_root_replay", dict(max_seconds=120))
    assert replay["view"]["episode_id"] == result["view"]["episode_id"]
    assert session.task.recipe_json == original.recipe_json
    with pytest.raises(ValueError, match="Generation failed"):
        session.request(
            "gp_root_query",
            dict(seed="47", task_id=str(2**55 + 11), locations=None, max_seconds=120),
        )
    assert session.task.recipe_json == original.recipe_json


@pytest.mark.parametrize("change", ["ancestor", "locations", "extra", "identity", "cap"])
def test_query_controls_reject_before_source_covariance(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, change: str
):
    fit = g.GraphConstruction(json.dumps(controls(tmp_path)), 0).build(
        p.CausalLimits(), episode_seed=31, task_id=2**55 + 7
    )
    draft: dict[str, Any] = dict(source=asdict(fit), locations=None)
    seed, task_id = 47, 2**55 + 11
    cap = p.CausalLimits()
    if change == "ancestor":
        seed, task_id = 31, 2**55 + 7
    elif change == "locations":
        draft["locations"] = [[0]]
    elif change == "extra":
        draft["unused"] = 0
    elif change == "identity":
        seed = True
    else:
        cap = p.CausalLimits(working_bytes=1)

    def forbidden(*args: object, **kwargs: object):
        pytest.fail("invalid query controls reached numerical replay")

    monkeypatch.setattr(np.linalg, "cholesky", forbidden)
    monkeypatch.setattr(np.random, "default_rng", forbidden)
    with pytest.raises(ValueError, match="GP|limit|locations"):
        p.construct_gp_root_calibration_query(draft, seed=seed, task_id=task_id, limits=cap)


@pytest.mark.parametrize("change", ["valid", "source", "request", "identity", "execution"])
def test_query_guarded_producer_complete_custody(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, change: str
):
    from dataclasses import replace

    fit = g.GraphConstruction(json.dumps(controls(tmp_path)), 0).build(
        p.CausalLimits(), episode_seed=31, task_id=2**55 + 7
    )
    draft = dict(topology="query-calibrated-gp-root", source=asdict(fit), locations=None)
    construction = g.GraphConstruction(json.dumps(draft), 0)
    query = construction.build(p.CausalLimits(), episode_seed=47, task_id=2**55 + 11)
    task = p.generate_task(query, seed=47, task_id=2**55 + 11)
    record = json.loads(task.recipe_json)
    if change == "source":
        record["recipe"]["source"]["fit"]["source"]["seed"] += 1
    elif change == "request":
        record["recipe"]["original"]["source"]["request"]["rows"] += 1
    elif change == "identity":
        record["seed"] += 1
    elif change == "execution":
        record["execution"]["python"] = ""

    def forbidden(*args: object, **kwargs: object):
        pytest.fail("known query producer metadata mismatch reached replay")

    if change != "valid":
        monkeypatch.setattr(np.linalg, "cholesky", forbidden)
    request = g.request_from_dict(
        dict(
            construction=dict(controls_json=construction.controls_json, seed=0),
            seed=47,
            start=2**55 + 11,
            count=1,
            max_seconds=120,
            max_bytes=10_000_000,
        )
    )
    result = g.run_generation(
        request, tmp_path / "run", produce=lambda _: replace(task, recipe_json=json.dumps(record))
    )
    assert result.completed == (1 if change == "valid" else 0)
    assert result.failed == (0 if change == "valid" else 1)
