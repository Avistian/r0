"""Topology families compose with typed mechanisms without losing their source receipt."""

import copy
import json
from dataclasses import asdict, replace
from typing import Any

import numpy as np
import pytest
from test_typed_parent_topology import composed_request

from r0.modules import priors as p
from r0.modules.priors.lib.generator_snapshot import GeneratorSnapshot
from r0.modules.priors.lib.structural_topology_envelope import SampledStructuralTopology


@pytest.mark.parametrize("family", ["functional", "native", "native-vector", "converted"])
@pytest.mark.parametrize(
    "prior",
    [
        p.CauchyGraphPrior(4, 1.0),
        p.AttachmentGraphPrior(4, edges_per_node=2),
        p.MotifGraphPrior(1, rewrite_steps=2),
    ],
)
@pytest.mark.parametrize(
    "engine",
    [np.random.PCG64, np.random.PCG64DXSM, np.random.Philox, np.random.SFC64, np.random.MT19937],
)
def test_structural_topology_typed_replay(
    family: str,
    prior: p.CauchyGraphPrior | p.AttachmentGraphPrior | p.MotifGraphPrior,
    engine: type[Any],
):
    request = replace(
        composed_request(family, "extension", "sum"), topology=p.StructuralTopologyRequest(prior)
    )
    streams = [np.random.Generator(engine(141 + i)) for i in range(5)]
    result = p.sample_typed_topology_graph(
        request,
        topology_rng=streams[0],
        named_rng=streams[1],
        raw_rng=streams[2],
        calibration_rng=streams[3],
        fit_rng=streams[4],
    )
    assert isinstance(result.topology, SampledStructuralTopology)
    assert result.topology.graph.prior == prior
    baseline_rng = np.random.Generator(engine(141))
    baseline = (
        p.sample_cauchy_graph(prior, rng=baseline_rng, limits=request.limits)
        if isinstance(prior, p.CauchyGraphPrior)
        else p.sample_structural_graph(
            prior, rng=baseline_rng, limits=request.limits, raw_limits=request.raw_limits
        )
    )
    assert asdict(baseline) == asdict(result.topology.graph)
    assert asdict(
        p.typed_topology_graph_from_dict(
            json.loads(json.dumps(asdict(result))), limits=request.limits
        )
    ) == asdict(result)
    task = p.generate_task(result, seed=211, task_id=41)
    trace = p.explain_generation(task, [7, 2, 7], limits=request.limits)
    assert trace["rows"] == [7, 2, 7]
    assert "P2" not in task.mechanism_json


@pytest.mark.parametrize("field", ["prior", "parents", "state", "limits"])
def test_structural_bad_binding_rejects_before_restore(field: str, monkeypatch: pytest.MonkeyPatch):
    request = replace(
        composed_request("functional", "extension", "sum"),
        topology=p.StructuralTopologyRequest(p.AttachmentGraphPrior(4, edges_per_node=2)),
    )
    streams = [np.random.default_rng(141 + i) for i in range(5)]
    result = p.sample_typed_topology_graph(
        request,
        topology_rng=streams[0],
        named_rng=streams[1],
        raw_rng=streams[2],
        calibration_rng=streams[3],
        fit_rng=streams[4],
    )
    raw = copy.deepcopy(asdict(result))
    if field == "prior":
        raw["topology"]["graph"]["prior"]["power"] = 2.0
    elif field == "parents":
        raw["topology"]["graph"]["parents"] = ((), (), (), ())
    elif field == "state":
        raw["topology"]["final"] = raw["topology"]["initial"]
    else:
        raw["topology"]["limits"]["nodes"] += 1

    def forbidden(*args: Any, **kwargs: Any):
        raise AssertionError("restored RNG before structural metadata rejection")

    monkeypatch.setattr(GeneratorSnapshot, "restore", forbidden)
    with pytest.raises(ValueError, match="structural"):
        p.typed_topology_graph_from_dict(raw, limits=request.limits)


def test_structural_failure_rolls_back_all_streams(monkeypatch: pytest.MonkeyPatch):
    from r0.modules.priors.lib import typed_topology_graph as sampling

    request = replace(
        composed_request("functional", "extension", "sum"),
        topology=p.StructuralTopologyRequest(p.CauchyGraphPrior(4, 2.0)),
    )
    streams = [np.random.default_rng(141 + i) for i in range(5)]
    before = [copy.deepcopy(stream.bit_generator.state) for stream in streams]

    def fail(*args: Any, **kwargs: Any):
        raise ValueError("forced downstream mechanism failure")

    monkeypatch.setattr(sampling, "sample_geometry_graph", fail)
    with pytest.raises(ValueError, match="forced downstream"):
        p.sample_typed_topology_graph(
            request,
            topology_rng=streams[0],
            named_rng=streams[1],
            raw_rng=streams[2],
            calibration_rng=streams[3],
            fit_rng=streams[4],
        )
    for stream, state in zip(streams, before, strict=True):
        np.testing.assert_equal(stream.bit_generator.state, state)


@pytest.mark.parametrize("offset", [-1e100, 1e100])
def test_cauchy_extremes_are_not_retried(offset: float):
    request = replace(
        composed_request("native", "extension", "sum"),
        topology=p.StructuralTopologyRequest(p.CauchyGraphPrior(4, offset)),
    )
    streams = [np.random.default_rng(141 + i) for i in range(5)]
    result = p.sample_typed_topology_graph(
        request,
        topology_rng=streams[0],
        named_rng=streams[1],
        raw_rng=streams[2],
        calibration_rng=streams[3],
        fit_rng=streams[4],
    )
    assert result.topology.parents == tuple(tuple(range(i)) if offset > 0 else () for i in range(4))
    assert len(result.topology.raw) == 4
