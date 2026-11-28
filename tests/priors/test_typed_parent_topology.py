"""Retained multi-parent edges exercise source-specific composition in all families."""

from dataclasses import asdict, replace
from typing import Any

import numpy as np
import pytest
import torch
from numpy.typing import NDArray
from typed_topology_fixture import request

from r0.modules import priors as p


def composed_request(family: str, source: str, aggregation: str) -> p.TypedParentTopologyRequest:
    base = request(family)
    controls = {f: getattr(base, f) for f in base.__dataclass_fields__ if f != "version"}
    controls.update(
        topology=p.FixedParentTopology(((), (0,), (0, 1), (1, 2))),
        composition=p.ParentCompositionPrior(
            source=source,
            aggregation=aggregation,
            mode="sample" if source == "mitra2" else "aggregate",
        ),
        limits=p.CausalLimits(parameters=2_000_000_000, working_bytes=1024**4),
        raw_limits=p.RawDrawLimits(numbers=2_000_000_000, draws=2_000_000, working_bytes=1024**4),
    )
    return p.TypedParentTopologyRequest(**controls)


@pytest.mark.parametrize("family", ["functional", "native", "native-vector", "converted"])
@pytest.mark.parametrize("source", ["tabiclv2", "mitra2", "extension"])
@pytest.mark.parametrize(
    "engine",
    [np.random.PCG64, np.random.PCG64DXSM, np.random.Philox, np.random.SFC64, np.random.MT19937],
)
def test_typed_parent_real_multi_parent_replay(family: str, source: str, engine: type[Any]):
    composed = composed_request(family, source, "max")
    streams = [np.random.Generator(engine(141 + i)) for i in range(5)]
    before = streams[0].bit_generator.state
    sampled = p.sample_typed_topology_graph(
        composed,
        topology_rng=streams[0],
        named_rng=streams[1],
        raw_rng=streams[2],
        calibration_rng=streams[3],
        fit_rng=streams[4],
    )
    np.testing.assert_equal(streams[0].bit_generator.state, before)
    assert sampled.topology.parents == (((), (0,), (0, 1), (1, 2)))
    assert getattr(sampled.graph.request, "composition", None) == composed.composition
    assert asdict(
        p.typed_topology_graph_from_dict(asdict(sampled), limits=composed.limits)
    ) == asdict(sampled)
    task = p.generate_task(sampled, seed=211, task_id=41)
    assert p.explain_generation(task, [7, 2, 7], limits=composed.limits)["rows"] == [7, 2, 7]


_REDUCTIONS = [
    (source, option)
    for source in ("tabiclv2", "mitra2", "extension")
    for option in p.ParentCompositionPrior(source=source).choices
]


@pytest.mark.parametrize("family", ["functional", "native", "native-vector", "converted"])
@pytest.mark.parametrize(("source", "aggregation"), _REDUCTIONS)
def test_all_reductions_keep_actual_parent_members(family: str, source: str, aggregation: str):
    controls = composed_request(family, source, aggregation)
    streams = [np.random.default_rng(41 + i) for i in range(5)]
    sampled = p.sample_typed_topology_graph(
        controls,
        topology_rng=streams[0],
        named_rng=streams[1],
        raw_rng=streams[2],
        calibration_rng=streams[3],
        fit_rng=streams[4],
    )
    node = sampled.runtime.nodes[2]
    equation = node if isinstance(node, p.FunctionalCausalNode) else node.equation
    function = equation.function
    assert isinstance(function, p.FrozenParentFunction)
    assert function.aggregation == aggregation
    assert len(function.children) == 2
    assert (function.joint is not None) == (source == "mitra2")
    if aggregation == "mlp":
        assert function.reducer is not None
        assert function.reducer.input_width == (2 + int(source == "mitra2")) * function.output_width
        return
    x = np.linspace(-0.75, 0.9, 3 * function.input_width).reshape(3, function.input_width)
    members: list[NDArray[np.float64]] = []
    offset = 0
    if function.joint is not None:
        assert isinstance(function.joint, p.FrozenLinearFunction)
        members.append(x @ np.array(function.joint.matrix).T)
    for child in function.children:
        assert isinstance(child, p.FrozenLinearFunction)
        members.append(x[:, offset : offset + child.input_width] @ np.array(child.matrix).T)
        offset += child.input_width
    stacked = np.stack(members)
    if aggregation == "sum":
        expected = stacked.sum(axis=0)
    elif aggregation == "product":
        expected = stacked.prod(axis=0)
    elif aggregation == "mean":
        expected = stacked.mean(axis=0)
    elif aggregation == "max":
        expected = stacked.max(axis=0)
    elif aggregation == "logsumexp":
        expected = np.logaddexp.reduce(stacked, axis=0)
    else:
        expected = np.einsum("i,ijk->jk", np.array(function.weights), stacked)
    actual = p.evaluate_frozen_function(function, torch.tensor(x)).numpy()
    np.testing.assert_allclose(actual, expected, atol=1e-12, rtol=1e-12)


@pytest.mark.parametrize("field", ["composition", "version", "parents", "final", "working"])
def test_invalid_parent_metadata_rejects_before_restore(
    field: str, monkeypatch: pytest.MonkeyPatch
):
    from r0.modules.priors.lib.generator_snapshot import GeneratorSnapshot

    controls = composed_request("functional", "tabiclv2", "max")
    streams = [np.random.default_rng(i) for i in range(5)]
    sampled = p.sample_typed_topology_graph(
        controls,
        topology_rng=streams[0],
        named_rng=streams[1],
        raw_rng=streams[2],
        calibration_rng=streams[3],
        fit_rng=streams[4],
    )
    raw = asdict(sampled)
    if field == "composition":
        raw["request"]["composition"]["aggregation"] = "unknown"
    elif field == "version":
        raw["version"] = "sampled-typed-topology-r0-v1"
    elif field == "parents":
        raw["topology"]["prior"]["parents"] = ((), (), (), ())
    elif field == "final":
        raw["final"] = (*raw["final"][:1], *raw["initial"][1:])
    calls: list[int] = []

    def fail(*args: Any, **kwargs: Any):
        calls.append(1)
        raise AssertionError("early restore")

    monkeypatch.setattr(GeneratorSnapshot, "restore", fail)
    with pytest.raises(ValueError, match="composition|version|receipt|stream|limit|budget"):
        p.typed_topology_graph_from_dict(
            raw,
            limits=replace(controls.limits, working_bytes=1)
            if field == "working"
            else controls.limits,
        )
    assert calls == []


def test_neural_reducer_support_is_reserved_even_for_linear_edges():
    linear = composed_request("functional", "extension", "max")
    neural = replace(
        linear,
        composition=p.ParentCompositionPrior(
            source="extension", mode="aggregate", aggregation="mlp"
        ),
    )
    linear_cost, neural_cost = linear.admit(), neural.admit()
    assert neural_cost["parameters"] > linear_cost["parameters"]
    assert neural_cost["estimated_working_bytes"] > linear_cost["estimated_working_bytes"]


def test_parent_materialization_failure_rolls_back_every_stream(monkeypatch: pytest.MonkeyPatch):
    import copy

    controls = composed_request("functional", "extension", "weighted")
    streams = [np.random.default_rng(212 + i) for i in range(5)]
    before = copy.deepcopy([s.bit_generator.state for s in streams])

    def fail(*args: Any, **kwargs: Any):
        raise ValueError("deliberate parent materialization failure")

    monkeypatch.setattr(p.FrozenParentFunction, "__post_init__", fail)
    with pytest.raises(ValueError, match="deliberate parent"):
        p.sample_typed_topology_graph(
            controls,
            topology_rng=streams[0],
            named_rng=streams[1],
            raw_rng=streams[2],
            calibration_rng=streams[3],
            fit_rng=streams[4],
        )
    assert [s.bit_generator.state for s in streams] == before
