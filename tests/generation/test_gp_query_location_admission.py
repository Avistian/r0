"""Raw location admission precedes copies and keeps fixed-coordinate custody."""

import gc
import json
import tracemalloc
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from r0.modules import generation as g
from r0.modules import priors as p
from r0.modules.explorer import ExplorerSession


@pytest.fixture(scope="module")
def fitted(tmp_path_factory: pytest.TempPathFactory):
    catalog = ExplorerSession(tmp_path_factory.mktemp("gp-locations")).request(
        "construction_examples", {}
    )["examples"]
    draft = catalog["GP root DAG: functional linear / regression"]
    draft["request"].update(
        rows=16, dimension=1, location_family="fixed-r0", locations=[[float(i)] for i in range(16)]
    )
    source = g.GraphConstruction(json.dumps(draft), 0).build(
        p.CausalLimits(), episode_seed=31, task_id=41
    )
    fit = g.GraphConstruction(
        json.dumps(
            dict(
                topology="calibrate-gp-root",
                source=asdict(source),
                calibration_rows=32,
                calibration_seed=43,
                calibration_task_id=47,
                calibration_locations=[[float(i)] for i in range(32)],
            )
        ),
        0,
    ).build(p.CausalLimits(), episode_seed=31, task_id=41)
    query = p.construct_gp_root_calibration_query(
        dict(source=asdict(fit), locations=None), seed=53, task_id=59, limits=p.CausalLimits()
    )
    task = p.generate_task(query, seed=53, task_id=59)
    return asdict(fit), json.loads(task.recipe_json)


@pytest.mark.parametrize("producer", [False, True])
def test_tiny_caller_cap_rejects_before_large_location_copy(fitted: Any, producer: bool):
    source, record = fitted
    controls = dict(source=source, locations=[[0.0] for _ in range(200_000)])
    gc.collect()
    tracemalloc.start()
    try:
        if producer:
            assert not p.gp_query_construction_matches(
                dict(topology="query-calibrated-gp-root", **controls),
                record,
                seed=53,
                task_id=59,
                limits=p.CausalLimits(working_bytes=1),
            )
        else:
            with pytest.raises(ValueError, match="limit"):
                p.construct_gp_root_calibration_query(
                    controls, seed=53, task_id=59, limits=p.CausalLimits(working_bytes=1)
                )
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    assert peak < 1024**2, "known rejection copied the caller-owned oversized location matrix"


@pytest.mark.parametrize("mutation", ["rows", "width", "bool", "nan", "inf", "text", "huge"])
def test_raw_location_geometry_rejects_before_replay(
    fitted: Any, monkeypatch: pytest.MonkeyPatch, mutation: str
):
    source, _ = fitted
    locations: list[Any] = [[float(i)] for i in range(16)]
    if mutation == "rows":
        locations.append([0.0])
    elif mutation == "width":
        locations[0] = [0.0, 1.0]
    else:
        locations[0] = [
            {"bool": True, "nan": float("nan"), "inf": float("inf"), "text": "0", "huge": 10**500}[
                mutation
            ]
        ]

    def forbidden(*args: Any, **kwargs: Any):
        pytest.fail("invalid location metadata reached covariance or random work")

    monkeypatch.setattr(np.linalg, "cholesky", forbidden)
    monkeypatch.setattr(np.random, "default_rng", forbidden)
    with pytest.raises(ValueError, match="location"):
        p.construct_gp_root_calibration_query(
            dict(source=source, locations=locations), seed=53, task_id=59, limits=p.CausalLimits()
        )


def test_fixed_list_and_tuple_locations_keep_identical_artifacts(fitted: Any, tmp_path: Path):
    source, _ = fitted
    rows = [[float(i) + 0.125] for i in range(16)]
    results = [
        p.construct_gp_root_calibration_query(
            dict(source=source, locations=locations), seed=53, task_id=59, limits=p.CausalLimits()
        )
        for locations in (rows, tuple(tuple(row) for row in rows))
    ]
    assert asdict(results[0]) == asdict(results[1])
    assert asdict(results[0].source) == source
    tasks = [p.generate_task(result, seed=53, task_id=59) for result in results]
    assert tasks[0].recipe_json == tasks[1].recipe_json
    for index, task in enumerate(tasks):
        path = tmp_path / f"query-{index}.zip"
        g.save_group(path, [(59, task)], max_bytes=64 * 1024**2)
        assert next(g.load_group(path)).recipe_json == task.recipe_json
