"""Converter plans occupy reserved numerical channels, never latent channels."""

import numpy as np
import pytest

from r0.modules import priors


@pytest.mark.parametrize(("low", "high"), [(1, 3), (3, 5)])
def test_pipeline_group_assignment_uses_half_open_spans(low: int, high: int) -> None:
    functions = priors.JointFunctionPrior(
        functions="lin", matrix=priors.JointMatrixPrior(families=("gaussian",))
    )
    request = priors.GeometryPipelineRequest(
        topology=priors.MotifGraphPrior(depth=1),
        task=priors.LinearRecipe(
            rows=16, context_rows=8, features=2, signal_features=2, active_features=2
        ),
        width_prior=priors.JointWidthPrior(low=low, high=high),
        calibration_rows=4,
        root_prior=priors.JointRootPrior(families=("gaussian",), function=functions),
        function_prior=functions,
        pipeline_prior=priors.JointPipelinePrior(
            weight_family="uniform",
            gaussian_noise=True,
            converter=priors.JointNumericalConverterPrior(
                variant="historical-tuple", allow_activation=False
            ),
        ),
    )
    streams = [np.random.default_rng(s) for s in np.random.SeedSequence(28).spawn(4)]
    graph = priors.sample_geometry_graph(
        request,
        named_rng=streams[0],
        raw_rng=streams[1],
        calibration_rng=streams[2],
        fit_rng=streams[3],
    )
    assert isinstance(graph, priors.SampledPipelineGeometry)
    expected: list[list[str]] = [[] for _ in graph.runtime.nodes]
    # Independent oracle uses actual feature assignments, without reading spans.
    for group, assignments in zip(request.groups, graph.observations.assignments, strict=True):
        for node in assignments:
            expected[node].append(group.name)
    assert max(map(len, expected)) > 1
    for index, fitted in enumerate(graph.pipelines):
        assert [c.feature_group for c in fitted.plan.converters] == expected[index]
        assert len(fitted.pipeline.converters) == graph.layout.converter_channels[index]
        assert [c.channel for c in fitted.pipeline.converters] == list(range(len(expected[index])))
        assert fitted.pipeline.channels > len(expected[index])
    assert graph.replay() == graph
    task = priors.generate_task(graph, seed=17, task_id=2)
    assert priors.explain_generation(task, [7, 0, 7])["rows"] == [7, 0, 7]
