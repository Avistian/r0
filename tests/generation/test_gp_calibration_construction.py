"""Fit a separate finite GP population inside guarded construction."""

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from r0.modules import generation as g
from r0.modules import priors as p
from r0.modules.explorer import ExplorerSession


def controls(tmp_path: Path, family: str = "functional") -> dict[str, Any]:
    catalog = ExplorerSession(tmp_path).request("construction_examples", {})["examples"]
    source = g.GraphConstruction(
        json.dumps(catalog[f"GP root DAG: {family} linear / regression"]), 17
    ).build(p.CausalLimits(), episode_seed=31, task_id=2**55 + 7)
    return dict(
        topology="calibrate-gp-root",
        source=asdict(source),
        calibration_rows=32,
        calibration_seed=41,
        calibration_task_id=2**55 + 9,
        calibration_locations=None,
    )


@pytest.mark.parametrize("family", ["functional", "native", "vector", "converted"])
def test_gp_calibration_constructor_literal(tmp_path: Path, family: str):
    draft = controls(tmp_path, family)
    construction = g.GraphConstruction(json.dumps(draft), 17)
    assert construction.task_record()["rows"] == 16
    actual = construction.build(p.CausalLimits(), episode_seed=31, task_id=2**55 + 7)
    assert isinstance(actual, p.GPRootCalibration)
    fit = np.asarray(actual.fit.source.draw.values)
    raw = 2 * fit[:, 0] - fit[:, 1]
    np.testing.assert_allclose(actual.calibration.center, [raw.mean()], atol=1e-12)
    np.testing.assert_allclose(actual.calibration.scale, [raw.std()], atol=1e-12)
    assert actual.original.source.task_id == 2**55 + 7
    assert actual.fit.source.task_id == 2**55 + 9
    task = p.generate_task(actual, seed=31, task_id=2**55 + 7)
    assert p.explain_generation(task, [15, 1, 15])["rows"] == [15, 1, 15]


@pytest.mark.parametrize("change", ["rows", "identity", "boolean", "caps", "source_identity"])
def test_gp_fit_constructor_preflight_before_covariance(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, change: str
):
    draft = controls(tmp_path)
    if change == "rows":
        draft["calibration_rows"] = 16
    elif change == "identity":
        draft.update(calibration_seed=31, calibration_task_id=2**55 + 7)
    elif change == "boolean":
        draft["calibration_seed"] = True
    elif change == "source_identity":
        draft["source"]["source"]["task_id"] += 1
    cap = p.CausalLimits(working_bytes=120 * 1024**2) if change == "caps" else p.CausalLimits()

    def forbidden(*args: object, **kwargs: object):
        pytest.fail("known GP fitting metadata error reached covariance work")

    monkeypatch.setattr(np.linalg, "cholesky", forbidden)
    with pytest.raises(ValueError, match="GP"):
        g.GraphConstruction(json.dumps(draft), 17).build(cap, episode_seed=31, task_id=2**55 + 7)


def test_guarded_gp_fit_and_failure_preserve_source(tmp_path: Path):
    session = ExplorerSession(tmp_path)
    draft = controls(tmp_path / "catalog")
    source = draft["source"]
    original = session.request(
        "generate",
        {"recipe": source, "seed": 31, "task_id": str(2**55 + 7), "max_seconds": 120},
    )
    with pytest.raises(ValueError, match="Generation failed"):
        session.request(
            "gp_root_calibrate",
            dict(calibration_rows=32, calibration_seed="31", calibration_task_id=str(2**55 + 7)),
        )
    assert session.request("view", {})["view"]["episode_id"] == original["view"]["episode_id"]
    fitted = session.request(
        "gp_root_calibrate",
        dict(
            calibration_rows=32,
            calibration_seed="41",
            calibration_task_id=str(2**55 + 9),
            max_seconds=120,
        ),
    )
    recipe = fitted["view"]["recipe"]["recipe"]
    assert recipe["version"] == "gp-root-calibration-v1"
    assert int(recipe["fit"]["source"]["task_id"]) == 2**55 + 9
    replay = session.request("gp_root_replay", {"max_seconds": 120})
    assert replay["view"]["episode_id"] == fitted["view"]["episode_id"]


@pytest.mark.parametrize("change", ["valid", "seed", "fit_identity", "fit_geometry"])
def test_gp_fit_producer_metadata_binding(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, change: str
):
    from dataclasses import replace

    draft = controls(tmp_path / "catalog")
    construction = g.GraphConstruction(json.dumps(draft), 17)
    recipe = construction.build(p.CausalLimits(), episode_seed=31, task_id=2**55 + 7)
    task = p.generate_task(recipe, seed=31, task_id=2**55 + 7)
    record = json.loads(task.recipe_json)
    if change == "seed":
        record["seed"] = 32
    elif change == "fit_identity":
        record["recipe"]["fit"]["source"]["task_id"] += 1
    elif change == "fit_geometry":
        record["recipe"]["fit"]["source"]["request"]["rows"] = 33
    candidate = replace(task, recipe_json=json.dumps(record))

    def forbidden(*args: object, **kwargs: object):
        pytest.fail("known producer fit metadata mismatch reached covariance work")

    if change != "valid":
        monkeypatch.setattr(np.linalg, "cholesky", forbidden)
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
    result = g.run_generation(request, tmp_path / "run", produce=lambda _: candidate)
    assert result.completed == (1 if change == "valid" else 0)
    assert result.failed == (0 if change == "valid" else 1)


def test_raw_gp_fit_complete_aggregate_rejects_before_decode(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    cap = p.CausalLimits(working_bytes=120 * 1024**2)
    catalog = ExplorerSession(tmp_path).request("construction_examples", {})["examples"]
    source_controls = catalog["GP root DAG: functional linear / regression"]
    source_controls["request"].update(rows=32, limits=asdict(cap))
    source_controls["base"]["task"].update(rows=32, context_rows=16)
    source_controls["base"]["limits"] = asdict(cap)
    source = g.GraphConstruction(json.dumps(source_controls), 17).build(
        cap, episode_seed=31, task_id=2**55 + 7
    )
    draft = dict(
        topology="calibrate-gp-root",
        source=asdict(source),
        calibration_rows=32,
        calibration_seed=41,
        calibration_task_id=2**55 + 9,
        calibration_locations=None,
    )

    def forbidden(*args: object, **kwargs: object):
        pytest.fail("complete future fit aggregate rejection followed covariance work")

    monkeypatch.setattr(np.linalg, "cholesky", forbidden)
    with pytest.raises(ValueError, match="aggregate"):
        g.GraphConstruction(json.dumps(draft), 17).build(cap, episode_seed=31, task_id=2**55 + 7)


@pytest.mark.parametrize(("field", "value"), [("device", "cuda"), ("count", 2), ("seed", 32)])
def test_gp_fit_run_transport_rejects_before_work(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, field: str, value: Any
):
    draft = controls(tmp_path)
    request = dict(
        construction=dict(controls_json=json.dumps(draft), seed=17),
        seed=31,
        start=2**55 + 7,
        count=1,
        max_seconds=120,
        max_bytes=10_000_000,
    )
    request[field] = value

    def forbidden(*args: object, **kwargs: object):
        pytest.fail("known run transport mismatch reached covariance work")

    monkeypatch.setattr(np.linalg, "cholesky", forbidden)
    with pytest.raises(ValueError, match="exact retained source identity and CPU"):
        g.request_from_dict(request)


def test_gp_fit_explicit_fixed_locations(tmp_path: Path):
    catalog = ExplorerSession(tmp_path).request("construction_examples", {})["examples"]
    source_controls = catalog["GP root DAG: functional linear / regression"]
    source_controls["request"].update(
        location_family="fixed-r0", locations=[[i / 10, i / 7] for i in range(16)]
    )
    source = g.GraphConstruction(json.dumps(source_controls), 17).build(
        p.CausalLimits(), episode_seed=31, task_id=2**55 + 7
    )
    assert isinstance(source, p.GPRootDAGRecipe)
    locations = [[i / 13, i / 11] for i in range(32)]
    draft = dict(
        topology="calibrate-gp-root",
        source=asdict(source),
        calibration_rows=32,
        calibration_seed=41,
        calibration_task_id=2**55 + 9,
        calibration_locations=locations,
    )
    actual = g.GraphConstruction(json.dumps(draft), 17).build(
        p.CausalLimits(), episode_seed=31, task_id=2**55 + 7
    )
    assert isinstance(actual, p.GPRootCalibration)
    np.testing.assert_array_equal(actual.fit.source.draw.locations, locations)
    assert actual.original.source == source.source
