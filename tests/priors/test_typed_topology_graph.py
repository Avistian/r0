"""Four-family topology composition retains each original construction receipt."""

from dataclasses import asdict

import numpy as np
import pytest
from typed_topology_fixture import request

from r0.modules import priors as p


@pytest.mark.parametrize("family", ["functional", "native", "native-vector", "converted"])
def test_typed_topology_complete_replay(family: str):
    controls = request(family)
    streams = [np.random.default_rng(130 + i) for i in range(5)]
    sampled = p.sample_typed_topology_graph(
        controls,
        topology_rng=streams[0],
        named_rng=streams[1],
        raw_rng=streams[2],
        calibration_rng=streams[3],
        fit_rng=streams[4],
    )
    assert (
        tuple(
            n.parents if isinstance(n, p.FunctionalCausalNode) else n.equation.parents
            for n in sampled.runtime.nodes
        )
        == sampled.topology.parents
    )
    reopened = p.typed_topology_graph_from_dict(asdict(sampled), limits=controls.limits)
    assert asdict(reopened) == asdict(sampled)


def test_typed_topology_shared_generation_trace_dispatch():
    controls = request("native-vector")
    streams = [np.random.default_rng(730 + i) for i in range(5)]
    sampled = p.sample_typed_topology_graph(
        controls,
        topology_rng=streams[0],
        named_rng=streams[1],
        raw_rng=streams[2],
        calibration_rng=streams[3],
        fit_rng=streams[4],
    )
    task = p.generate_task(sampled, seed=751, task_id=2**55 + 701)
    trace = p.explain_generation(task, [9, 2, 9], limits=controls.limits)
    assert trace["rows"] == [9, 2, 9]
    assert any(n["id"] == "typed-topology-construction" for n in trace["nodes"])
    assert p.recipe_from_dict(asdict(sampled), limits=controls.limits) == sampled
