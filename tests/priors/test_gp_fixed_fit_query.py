"""New finite GP query populations retain one independent fit and complete history."""

import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pytest
from gp_root_fixture import gp_root_recipe
from numpy.typing import NDArray

from r0.modules import priors as p


@pytest.mark.parametrize("family", ["functional", "native", "vector", "converted"])
def test_fresh_gp_root_query_keeps_fixed_fit_and_complete_lineage(family: str):
    fit = p.calibrate_gp_root_dag(
        gp_root_recipe(family), calibration_rows=32, calibration_seed=101, calibration_task_id=103
    )
    query = p.realize_gp_root_calibration_query(fit, seed=2**55 + 1, task_id=2**55 + 3)
    assert asdict(query.source) == asdict(fit)
    assert query.fit == fit.fit
    assert query.calibration == fit.calibration
    assert query.original.base == fit.original.base
    assert query.original.bindings == fit.original.bindings
    assert query.original.source.population_id != fit.original.source.population_id
    values = np.asarray(query.original.source.draw.values)
    independent = np.asarray(fit.fit.source.draw.values)
    raw_fit = 2 * independent[:, 0] - independent[:, 1]
    expected = (2 * values[:, 0] - values[:, 1] - raw_fit.mean()) / raw_fit.std()
    task = p.generate_task(query, seed=2**55 + 1, task_id=2**55 + 3)
    np.testing.assert_allclose(task.targets, expected, atol=1e-12)
    assert p.recipe_from_dict(json.loads(json.dumps(asdict(query)))) == query
    next_query = p.realize_gp_root_calibration_query(query, seed=2**55 + 1, task_id=2**55 + 4)
    assert asdict(next_query.source) == asdict(query)
    assert next_query.calibration == fit.calibration


def test_query_task_archive_and_repeated_original_row_trace(tmp_path: Path):
    from r0.modules import generation as g

    fit = p.calibrate_gp_root_dag(
        gp_root_recipe("functional"),
        calibration_rows=32,
        calibration_seed=101,
        calibration_task_id=103,
    )
    query = p.realize_gp_root_calibration_query(fit, seed=211, task_id=2**55 + 7)
    task = p.generate_task(query, seed=211, task_id=2**55 + 7)
    path = tmp_path / "query.zip"
    g.save_group(path, [(2**55 + 7, task)], max_bytes=256 * 1024**2)
    loaded = next(g.load_group(path))
    assert loaded.recipe_json == task.recipe_json
    assert p.validate_gp_root_task(loaded)[0] == query
    trace = p.explain_generation(loaded, [15, 1, 15])
    assert trace["rows"] == [15, 1, 15]
    assert trace["fit_population_id"] == fit.fit.source.population_id
    assert trace["population_id"] == query.original.source.population_id
    nodes = {node["id"]: node for node in trace["nodes"]}
    np.testing.assert_allclose(nodes["signal"]["values"], task.scores[[15, 1, 15]])
    assert all(parent in nodes for node in nodes.values() for parent in node["parents"])
    fitting = p.explain_gp_calibration_fit(loaded, [31, 2, 31])
    assert fitting["population_role"] == "fit"
    assert fitting["episode_id"] == task.episode_id
    assert fitting["fit_task_id"] == 103
    assert fitting["total_population_rows"] == 32


def test_fixed_location_query_fails_closed_without_retries_or_source_changes(
    monkeypatch: pytest.MonkeyPatch,
):
    from dataclasses import replace

    original = gp_root_recipe("functional")
    locations = tuple((float(i) / 4.0,) for i in range(16))
    request = replace(
        original.source.request, dimension=1, location_family="fixed-r0", locations=locations
    )
    original = replace(original, source=p.realize_gp_columns(request, seed=47, task_id=59))
    fit = p.calibrate_gp_root_dag(
        original,
        calibration_rows=32,
        calibration_seed=101,
        calibration_task_id=103,
        calibration_locations=tuple((float(i) / 4.0,) for i in range(32)),
    )
    before = asdict(fit)
    cholesky = np.linalg.cholesky
    attempts: list[bool] = []

    def count_singular(matrix: NDArray[np.float64]):
        if np.array_equal(matrix, np.ones((16, 16))):
            attempts.append(True)
        return cholesky(matrix)

    monkeypatch.setattr(np.linalg, "cholesky", count_singular)
    with pytest.raises(ValueError, match="positive definite|Cholesky|singular"):
        p.realize_gp_root_calibration_query(fit, seed=211, task_id=223, locations=((0.0,),) * 16)
    assert len(attempts) == 1
    assert asdict(fit) == before


@pytest.mark.parametrize("location_family", ["fixed-r0", "grid-r0"])
def test_fixed_query_location_receipt_binding_rejects_before_covariance(
    monkeypatch: pytest.MonkeyPatch, location_family: str
):
    from dataclasses import replace

    original = gp_root_recipe("functional")
    request = replace(
        original.source.request,
        dimension=1,
        location_family=location_family,
        locations=tuple((float(i) / 4.0,) for i in range(16))
        if location_family == "fixed-r0"
        else (),
    )
    original = replace(original, source=p.realize_gp_columns(request, seed=47, task_id=59))
    fit = p.calibrate_gp_root_dag(
        original,
        calibration_rows=32,
        calibration_seed=101,
        calibration_task_id=103,
        calibration_locations=tuple((float(i) / 4.0,) for i in range(32))
        if location_family == "fixed-r0"
        else None,
    )
    query = p.realize_gp_root_calibration_query(fit, seed=211, task_id=223)
    raw = json.loads(json.dumps(asdict(query)))
    raw["original"]["source"]["draw"]["locations"][0][0] = 0.25

    def forbidden(*args: object, **kwargs: object):
        pytest.fail("fixed coordinate metadata began covariance replay")

    monkeypatch.setattr(np.linalg, "cholesky", forbidden)
    with pytest.raises(ValueError, match="location|coordinate"):
        p.gp_root_calibration_query_from_dict(raw)
