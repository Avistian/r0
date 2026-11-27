"""Converter inputs occupy contiguous node prefixes before hidden latent channels."""

from dataclasses import replace

import numpy as np
import pytest

from r0.modules import priors


def test_channel_layout_reserves_each_feature_width_in_group_order() -> None:
    context = priors.JointSamplingContext(np.random.default_rng(44), mode="local")
    plan = priors.sample_joint_observations(
        context,
        "observations",
        nodes=3,
        groups=(priors.FeatureGroup("x", 4), priors.FeatureGroup("y", 2)),
        subsample=False,
    )
    observations = priors.materialize_joint_observations(
        plan, context.freeze(), rng=np.random.default_rng(92)
    )
    widths = ((1, 3, 2, 1), (4, 1))
    layout = priors.allocate_observation_channels(
        observations, context.freeze(), converter_widths=widths
    )
    expected: list[tuple[tuple[int, int, int], ...]] = []
    cursors = [0, 0, 0]
    for assignments, sizes in zip(observations.assignments, widths, strict=True):
        spans: list[tuple[int, int, int]] = []
        for node, size in zip(assignments, sizes, strict=True):
            spans.append((node, cursors[node], cursors[node] + size))
            cursors[node] += size
        expected.append(tuple(spans))
    assert layout.spans == tuple(expected)
    assert layout.converter_channels == tuple(cursors)
    layout.validate(context.freeze())
    sampled = priors.sample_joint_widths(
        context, "widths", converter_channels=layout.converter_channels
    )
    assert sampled.widths == tuple(c + h for c, h in zip(cursors, sampled.latent, strict=True))
    for group_spans in layout.spans:
        for node, start, stop in group_spans:
            assert 0 <= start < stop <= sampled.widths[node] - sampled.latent[node]
    changed = replace(layout, converter_channels=(0, 0, 0))
    with pytest.raises(ValueError, match="channel layout"):
        changed.validate(context.freeze())
    with pytest.raises(ValueError, match="converter widths"):
        priors.allocate_observation_channels(
            observations, context.freeze(), converter_widths=((1,), (1,))
        )
    with pytest.raises(ValueError, match="channel limit"):
        priors.allocate_observation_channels(
            observations, context.freeze(), converter_widths=((1024,) * 4, (1, 1))
        )


def test_numeric_channel_layout_drives_graph_readouts_and_keeps_hidden_channels() -> None:
    context = priors.JointSamplingContext(np.random.default_rng(44), mode="local")
    plan = priors.sample_joint_observations(
        context,
        "observations",
        nodes=3,
        groups=(priors.FeatureGroup("x", 4), priors.FeatureGroup("y", 1)),
        subsample=False,
    )
    observations = priors.materialize_joint_observations(
        plan, context.freeze(), rng=np.random.default_rng(92)
    )
    layout = priors.allocate_observation_channels(
        observations, context.freeze(), converter_widths=((1, 1, 1, 1), (1,))
    )
    widths = priors.sample_joint_widths(
        context,
        "widths",
        converter_channels=layout.converter_channels,
        prior=priors.JointWidthPrior(low=1, high=2),
    )
    functions = priors.JointFunctionPrior(
        functions="lin", matrix=priors.JointMatrixPrior(families=("gaussian",))
    )
    pipelines = tuple(
        priors.CausalNodePipeline(
            width, converters=tuple(priors.NumericalConverter(i) for i in range(count))
        )
        for width, count in zip(widths.widths, layout.converter_channels, strict=True)
    )
    request = priors.TopologyGraphRequest(
        topology=priors.CauchyGraphPrior(nodes=3, offset=1000.0),
        task=priors.LinearRecipe(
            rows=16, context_rows=8, features=4, signal_features=4, active_features=4
        ),
        widths=widths.widths,
        observed=tuple((node, start) for node, start, _ in layout.spans[0]),
        target=tuple((node, start) for node, start, _ in layout.spans[1]),
        calibration_rows=4,
        root_prior=priors.JointRootPrior(families=("gaussian",), function=functions),
        function_prior=functions,
        authored_pipelines=pipelines,
    )
    graph = priors.sample_topology_graph(
        request,
        named_rng=np.random.default_rng(114),
        raw_rng=np.random.default_rng(115),
        calibration_rng=np.random.default_rng(116),
        fit_rng=np.random.default_rng(117),
    )
    emitted = priors.generate_task(graph, seed=19, task_id=7)
    assert emitted.inputs.shape == (16, 4)
    assert priors.explain_generation(emitted, [7, 0, 7])["rows"] == [7, 0, 7]
    assert all(
        width - count == 1
        for width, count in zip(widths.widths, layout.converter_channels, strict=True)
    )
    assert graph.runtime.nodes[2].function.input_width == sum(widths.widths[:2])
