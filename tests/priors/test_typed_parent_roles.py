"""Composed mechanisms retain random sizes and genuinely typed classification roles."""

import math
from dataclasses import asdict, replace
from typing import Any

import numpy as np
import pytest
import torch
from test_typed_parent_topology import composed_request

from r0.modules import priors as p


def sample(controls: p.TypedParentTopologyRequest):
    streams = [np.random.default_rng(171 + i) for i in range(5)]
    return p.sample_typed_topology_graph(
        controls,
        topology_rng=streams[0],
        named_rng=streams[1],
        raw_rng=streams[2],
        calibration_rng=streams[3],
        fit_rng=streams[4],
    )


@pytest.mark.parametrize("family", ["functional", "native", "native-vector", "converted"])
@pytest.mark.parametrize("source", ["tabiclv2", "mitra2", "extension"])
def test_composition_preserves_random_topology_draw(family: str, source: str):
    controls = replace(
        composed_request(family, source, "max"),
        topology=p.TopologyHyperprior(log_nodes_low=math.log(3), log_nodes_high=math.log(7)),
    )
    graph = sample(controls)
    assert 3 <= len(graph.runtime.nodes) <= 7
    assert graph.initial[0] != graph.final[0]
    assert asdict(
        p.typed_topology_graph_from_dict(asdict(graph), limits=controls.limits)
    ) == asdict(graph)


@pytest.mark.parametrize("family", ["functional", "native", "native-vector", "converted"])
@pytest.mark.parametrize("classes", [2, 3])
@pytest.mark.parametrize("source", ["tabiclv2", "mitra2", "extension"])
def test_composed_classification_preserves_typed_roles(family: str, classes: int, source: str):
    controls = composed_request(family, source, "max")
    changes: dict[str, Any] = dict(
        task=replace(
            controls.task, task_type="binary" if classes == 2 else "multiclass", classes=classes
        )
    )
    if controls.native is not None:
        changes["native"] = replace(
            controls.native,
            hidden=p.NativeDomainPrior(kinds=("binary", "categorical"), cardinalities=(3,)),
            features=p.NativeDomainPrior(kinds=("binary",)),
            target=p.NativeDomainPrior(kinds=("binary",))
            if classes == 2
            else p.NativeDomainPrior(kinds=("categorical",), cardinalities=(classes,)),
        )
    if controls.converted is not None:
        changes["converted"] = replace(
            controls.converted,
            vocabularies=(("low", "high"), ()),
            target_vocabulary=tuple(str(i) for i in range(classes)),
        )
    controls = replace(controls, **changes)
    graph = sample(controls)
    restored = p.typed_topology_graph_from_dict(asdict(graph), limits=controls.limits)
    task = p.generate_task(restored, seed=71, task_id=19)
    assert task.targets.dtype == torch.int64
    assert torch.all((task.targets >= 0) & (task.targets < classes))
    assert torch.isfinite(task.inputs).all()
    assert asdict(restored) == asdict(graph)


@pytest.mark.parametrize("source", ["mitra2", "extension"])
@pytest.mark.parametrize("family", ["functional", "native", "native-vector", "converted"])
def test_sampled_neural_reducer_matches_independent_numpy(source: str, family: str):
    from typed_parent_oracle import neural_parent

    graph = sample(composed_request(family, source, "mlp"))
    node = graph.runtime.nodes[2]
    equation = node if isinstance(node, p.FunctionalCausalNode) else node.equation
    function = equation.function
    assert isinstance(function, p.FrozenParentFunction)
    inputs = np.linspace(-0.75, 0.9, 5 * function.input_width).reshape(5, function.input_width)
    expected = neural_parent(function, inputs)
    actual = p.evaluate_frozen_function(function, torch.tensor(inputs)).numpy()
    np.testing.assert_allclose(actual, expected, atol=1e-10, rtol=1e-10)
