"""Malformed GP query lineage and outer transport fail before numerical replay."""

import hashlib
import json
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from gp_root_fixture import gp_root_recipe

from r0.modules import generation as g
from r0.modules import priors as p


@pytest.fixture(scope="module")
def query():
    fit = p.calibrate_gp_root_dag(
        gp_root_recipe("functional"),
        calibration_rows=32,
        calibration_seed=101,
        calibration_task_id=103,
    )
    return p.realize_gp_root_calibration_query(fit, seed=211, task_id=223)


def forbid(monkeypatch: pytest.MonkeyPatch):
    def forbidden(*args: object, **kwargs: object):
        pytest.fail("metadata admission started GP covariance/RNG work")

    monkeypatch.setattr(np.linalg, "cholesky", forbidden)
    monkeypatch.setattr(np.random, "default_rng", forbidden)


@pytest.mark.parametrize(
    "case", ["original", "fit", "query", "bool", "overflow", "locations", "limits"]
)
def test_prospective_query_controls_precede_replay(
    query: p.GPRootCalibrationQuery, monkeypatch: pytest.MonkeyPatch, case: str
):
    controls: dict[str, Any] = dict(seed=307, task_id=311)
    if case in ("original", "fit", "query"):
        original = {"original": query.source.original, "fit": query.fit, "query": query.original}[
            case
        ]
        controls.update(seed=original.source.seed, task_id=original.source.task_id)
    elif case == "bool":
        controls["seed"] = True
    elif case == "overflow":
        controls["task_id"] = 2**63
    elif case == "locations":
        controls["locations"] = ((0.0,),)
    else:
        controls["limits"] = p.CausalLimits(working_bytes=1)
    forbid(monkeypatch)
    with pytest.raises(ValueError, match="GP|limit|metadata|locations|run|max_seconds"):
        p.realize_gp_root_calibration_query(query, **controls)


@pytest.mark.parametrize(
    "case", ["ancestor", "binding", "base", "rows", "limits", "extra", "version"]
)
def test_query_receipt_semantics_precede_replay(
    query: p.GPRootCalibrationQuery, monkeypatch: pytest.MonkeyPatch, case: str
):
    raw = json.loads(json.dumps(asdict(query)))
    if case == "ancestor":
        raw["original"]["source"]["seed"] = 101
        raw["original"]["source"]["task_id"] = 103
        raw["original"]["source"]["population_id"] = hashlib.sha256(
            json.dumps(
                (raw["original"]["source"]["request"], 101, 103),
                sort_keys=True,
                separators=(",", ":"),
            ).encode()
        ).hexdigest()
    elif case == "binding":
        raw["original"]["bindings"][0]["columns"] = [1, 0]
    elif case == "base":
        raw["original"]["base"]["nodes"][1]["function"]["matrix"] = [[3.0, -1.0]]
    elif case == "rows":
        raw["original"]["source"]["request"]["rows"] = 17
    elif case == "limits":
        raw["limits"]["working_bytes"] -= 1
    elif case == "extra":
        raw["discarded_history"] = []
    else:
        raw["version"] = "query-v0"
    forbid(monkeypatch)
    with pytest.raises(ValueError, match="GP|limit|metadata|locations|run|max_seconds"):
        p.gp_root_calibration_query_from_dict(raw)


@pytest.mark.parametrize(
    ("field", "value"), [("python", ""), ("numpy", ""), ("torch", ""), ("cpu_threads", 2**63)]
)
def test_query_execution_header_precedes_replay(
    query: p.GPRootCalibrationQuery, monkeypatch: pytest.MonkeyPatch, field: str, value: Any
):
    task = p.generate_task(query, seed=211, task_id=223)
    record = json.loads(task.recipe_json)
    record["execution"][field] = value
    malformed = replace(task, recipe_json=json.dumps(record))
    forbid(monkeypatch)
    with pytest.raises(ValueError, match="GP|limit|metadata|locations|run|max_seconds"):
        p.validate_gp_root_task(malformed)


@pytest.mark.parametrize(
    ("field", "value"),
    [("count", 2), ("seed", True), ("start", 224), ("max_seconds", 0), ("device", "cuda")],
)
def test_raw_query_run_header_precedes_replay(
    query: p.GPRootCalibrationQuery, monkeypatch: pytest.MonkeyPatch, field: str, value: Any
):
    raw = dict(
        recipe=asdict(query), seed=211, start=223, count=1, max_bytes=16 * 1024**2, max_seconds=60
    )
    raw[field] = value
    forbid(monkeypatch)
    with pytest.raises(ValueError, match="GP|limit|metadata|locations|run|max_seconds"):
        g.request_from_dict(raw)


@pytest.mark.parametrize("case", ["identity", "execution", "values"])
def test_query_producer_preserves_complete_saved_task_custody(
    query: p.GPRootCalibrationQuery, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, case: str
):
    task = p.generate_task(query, seed=211, task_id=223)
    request = g.RunRequest(
        query, seed=211, start=223, count=1, max_seconds=60, max_bytes=16 * 1024**2
    )
    if case == "values":
        malformed = replace(task, targets=task.targets + 1)
    else:
        record = json.loads(task.recipe_json)
        if case == "identity":
            record["seed"] = 212
        else:
            record["execution"]["numpy"] = ""
        malformed = replace(task, recipe_json=json.dumps(record))
        forbid(monkeypatch)
    summary = g.run_generation(request, tmp_path / case, produce=lambda _: malformed)
    assert summary.failed == 1
    assert summary.completed == 0
