"""Guarded GP-root construction preserves authored graphs and population identities."""

import json
from pathlib import Path

import numpy as np
import pytest

from r0.modules import generation as g
from r0.modules import priors as p
from r0.modules.explorer import ExplorerSession


def test_gp_root_catalog_covers_runtime_and_root_mechanism_variety(tmp_path: Path):
    catalog = ExplorerSession(tmp_path).request("construction_examples", {})["examples"]
    examples = {
        name: controls for name, controls in catalog.items() if name.startswith("GP root DAG:")
    }
    assert len(examples) >= 16
    seen: set[str] = set()
    for controls in examples.values():
        graph = g.GraphConstruction(json.dumps(controls), 17).build(
            p.CausalLimits(),
            episode_seed=31,
            task_id=2**55 + 7,
        )
        assert isinstance(graph, p.GPRootDAGRecipe)
        seen.add(graph.base.version)
        task = p.generate_task(graph, seed=31, task_id=2**55 + 7)
        assert p.explain_generation(task, [15, 1, 15])["rows"] == [15, 1, 15]
        assert graph.source.task_id == 2**55 + 7
    assert seen == {
        "functional-causal-v1",
        "native-causal-v1",
        "native-vector-causal-v1",
        "converted-causal-v1",
    }


@pytest.mark.parametrize("family", ["functional", "native", "vector", "converted"])
def test_gp_root_linear_construction_literal(tmp_path: Path, family: str):
    catalog = ExplorerSession(tmp_path).request("construction_examples", {})["examples"]
    controls = next(
        v for k, v in catalog.items() if k.startswith(f"GP root DAG: {family} linear / regression")
    )
    graph = g.GraphConstruction(json.dumps(controls), 17).build(
        p.CausalLimits(),
        episode_seed=31,
        task_id=2**55 + 7,
    )
    assert isinstance(graph, p.GPRootDAGRecipe)
    task = p.generate_task(graph, seed=31, task_id=2**55 + 7)
    raw = np.asarray(graph.source.draw.values)
    np.testing.assert_array_equal(task.inputs, raw)
    np.testing.assert_allclose(task.targets, 2 * raw[:, 0] - raw[:, 1], atol=1e-14)
    trace = p.explain_generation(task, [15, 1, 15])
    locations = np.asarray(graph.source.draw.locations)
    distance = ((locations[:, None, :] - locations[None, :, :]) ** 2).sum(axis=2)
    for j, matrix in enumerate(trace["shared_matrices"]):
        assert matrix["row_ids"] == matrix["column_row_ids"] == list(range(16))
        kernel = graph.source.request.kernels[j]
        covariance = kernel.amplitude**2 * np.exp(-distance / (2 * kernel.length_scale**2))
        np.testing.assert_allclose(matrix["covariance"], covariance, atol=1e-14)
        factor = np.asarray(matrix["factor"])
        np.testing.assert_allclose(factor @ factor.T, covariance, atol=1e-14)


@pytest.mark.parametrize("family", ["functional", "native", "vector", "converted"])
def test_gp_root_guarded_session_replay_and_draft(tmp_path: Path, family: str):
    session = ExplorerSession(tmp_path)
    catalog = session.request("construction_examples", {})["examples"]
    controls = next(
        v for k, v in catalog.items() if k.startswith(f"GP root DAG: {family} Conv / regression")
    )
    first = session.request(
        "construct",
        {
            "controls": controls,
            "construction_seed": "17",
            "seed": "31",
            "task_id": str(2**55 + 7),
        },
    )
    original = session.task
    assert original is not None
    trace = session.request("graph", {"indices": [15, 1, 15]})
    assert trace["graph"]["rows"] == [15, 1, 15]
    info = session.request("gp_root_controls", {})
    assert info["gp_root_dag"] is True
    assert info["controls"] == json.loads(json.dumps(controls))
    replay = session.request("gp_root_replay", {})
    assert replay["view"]["episode_id"] == first["view"]["episode_id"]
    assert session.task is not None
    np.testing.assert_array_equal(session.task.inputs, original.inputs)
    np.testing.assert_array_equal(session.task.targets, original.targets)
    second = session.request(
        "construct",
        {
            "controls": info["controls"],
            "construction_seed": "17",
            "seed": "31",
            "task_id": str(2**55 + 8),
        },
    )
    assert second["view"]["episode_id"] != first["view"]["episode_id"]


def test_gp_root_construction_custom_producer_cannot_forge_targets(tmp_path: Path):
    from dataclasses import replace

    controls = next(
        v
        for k, v in ExplorerSession(tmp_path / "catalog")
        .request("construction_examples", {})["examples"]
        .items()
        if k.startswith("GP root DAG: functional linear / regression")
    )
    construction = g.GraphConstruction(json.dumps(controls), 17)
    graph = construction.build(p.CausalLimits(), episode_seed=31, task_id=2**55 + 7)
    task = p.generate_task(graph, seed=31, task_id=2**55 + 7)
    targets = task.targets.clone()
    targets[0] += 1
    forged = replace(task, targets=targets)
    request = g.request_from_dict(
        {
            "construction": {"controls_json": construction.controls_json, "seed": 17},
            "seed": 31,
            "start": 2**55 + 7,
            "count": 1,
            "max_seconds": 30,
            "max_bytes": 10_000_000,
        }
    )
    result = g.run_generation(request, tmp_path / "run", produce=lambda _: forged)
    assert result.completed == 0
    assert result.failed == 1


def test_gp_root_linear_trace_uses_guarded_worker(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    session = ExplorerSession(tmp_path)
    controls = next(
        v
        for k, v in session.request("construction_examples", {})["examples"].items()
        if k.startswith("GP root DAG: functional linear / regression")
    )
    session.request(
        "construct",
        {
            "controls": controls,
            "construction_seed": "17",
            "seed": "31",
            "task_id": "59",
        },
    )

    def forbidden(*args: object, **kwargs: object):
        pytest.fail("GP source trace ran in the parent process")

    monkeypatch.setattr(p, "explain_generation", forbidden)
    trace = session.request("graph", {"indices": [15, 1, 15]})
    assert trace["graph"]["kind"] == "gp_root_dag"
