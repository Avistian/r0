"""Construction caps are identity, beyond whether a candidate fits those caps."""

import json
from copy import deepcopy
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from r0.modules import generation as g
from r0.modules import priors as p
from r0.modules.explorer import ExplorerSession


@pytest.fixture(scope="module")
def fit_records(tmp_path_factory: pytest.TempPathFactory):
    folder = tmp_path_factory.mktemp("gp-fit-caps")
    catalog = ExplorerSession(folder).request("construction_examples", {})["examples"]
    caps = p.CausalLimits(working_bytes=128 * 1024**2)
    output: dict[str, Any] = {}
    for family in ("functional", "native", "vector", "converted"):
        controls = deepcopy(catalog[f"GP root DAG: {family} linear / regression"])
        controls["base"]["limits"] = asdict(caps)
        controls["request"]["limits"] = asdict(caps)
        source = g.GraphConstruction(json.dumps(controls), 17).build(
            caps, episode_seed=31, task_id=2**55 + 7
        )
        draft = dict(
            topology="calibrate-gp-root",
            source=asdict(source),
            calibration_rows=32,
            calibration_seed=41,
            calibration_task_id=2**55 + 9,
            calibration_locations=None,
        )
        construction = g.GraphConstruction(json.dumps(draft), 17)
        fitted = construction.build(p.CausalLimits(), episode_seed=31, task_id=2**55 + 7)
        task = p.generate_task(fitted, seed=31, task_id=2**55 + 7)
        request = g.request_from_dict(
            dict(
                construction=dict(controls_json=construction.controls_json, seed=17),
                seed=31,
                start=2**55 + 7,
                count=1,
                max_seconds=120,
                max_bytes=10_000_000,
            )
        )
        output[family] = draft, task, request
    return output


@pytest.mark.parametrize("family", ["functional", "native", "vector", "converted"])
@pytest.mark.parametrize("case", ["outer", "fit", "both", "cpu"])
def test_gp_fit_producer_caps_before_covariance(
    fit_records: dict[str, Any],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    family: str,
    case: str,
):
    draft, task, request = fit_records[family]
    record = json.loads(task.recipe_json)
    caps = asdict(p.CausalLimits(working_bytes=128 * 1024**2))
    if case in ("outer", "both"):
        record["recipe"]["limits"] = caps
    if case in ("fit", "both"):
        record["recipe"]["fit"]["limits"] = caps
    if case == "cpu":
        record["execution"]["device"] = "cuda"
    candidate = replace(task, recipe_json=json.dumps(record))

    def forbidden(*args: Any, **kwargs: Any):
        pytest.fail("known GP fit producer metadata reached Cholesky")

    monkeypatch.setattr(np.linalg, "cholesky", forbidden)
    assert not p.gp_calibration_construction_matches(
        draft, record, seed=31, task_id=2**55 + 7, limits=p.CausalLimits()
    )
    result = g.run_generation(request, tmp_path, produce=lambda _: candidate)
    assert result.completed == 0
    assert result.failed == 1


@pytest.mark.parametrize(
    "case", ["fit_rows", "fit_seed", "fit_graph", "fit_bindings", "stream", "prefix"]
)
def test_gp_fit_other_known_metadata_before_covariance(
    fit_records: dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
    case: str,
):
    draft, task, _ = fit_records["functional"]
    record = json.loads(task.recipe_json)
    raw = record["recipe"]
    if case == "fit_rows":
        raw["fit"]["source"]["request"]["rows"] = 40
    elif case == "fit_seed":
        raw["fit"]["source"]["seed"] += 1
    elif case == "fit_graph":
        raw["fit"]["base"]["nodes"][1]["function"]["matrix"][0][0] = 7.0
    elif case == "fit_bindings":
        raw["fit"]["bindings"][0]["columns"] = [1, 0]
    elif case == "stream":
        raw["calibration"]["stream"] = "wrong"
    else:
        raw["calibration"]["prefix_fingerprint"] = "0" * 64

    def forbidden(*args: Any, **kwargs: Any):
        pytest.fail("known GP fit producer metadata reached Cholesky")

    monkeypatch.setattr(np.linalg, "cholesky", forbidden)
    assert not p.gp_calibration_construction_matches(
        draft, record, seed=31, task_id=2**55 + 7, limits=p.CausalLimits()
    )


@pytest.mark.parametrize("family", ["functional", "native", "vector", "converted"])
def test_gp_fit_distinct_valid_source_and_caller_caps(
    fit_records: dict[str, Any],
    tmp_path: Path,
    family: str,
):
    draft, task, request = fit_records[family]
    before = task.recipe_json
    assert p.gp_calibration_construction_matches(
        draft, json.loads(before), seed=31, task_id=2**55 + 7, limits=p.CausalLimits()
    )
    result = g.run_generation(request, tmp_path, produce=lambda _: task)
    assert result.completed == 1
    assert result.failed == 0
    restored = next(g.load_group(next(tmp_path.rglob("data-*.zip"))))
    assert restored.recipe_json == before
