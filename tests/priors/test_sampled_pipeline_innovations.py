"""Sampled innovations retain complete laws rather than a Gaussian-only scale."""

import json
import pickle
from dataclasses import asdict, fields, replace

import numpy as np
import pytest

from r0.modules import priors as p


def request_for(family: str) -> p.GeometryPipelineRequest:
    functions = p.JointFunctionPrior(
        functions="lin", matrix=p.JointMatrixPrior(families=("gaussian",))
    )
    return p.GeometryPipelineRequest(
        topology=p.AttachmentGraphPrior(nodes=3),
        task=p.LinearRecipe(
            rows=16, context_rows=8, features=2, signal_features=2, active_features=2
        ),
        subsample=False,
        width_prior=p.JointWidthPrior(low=1, high=3),
        calibration_rows=8,
        root_prior=p.JointRootPrior(families=("gaussian",), function=functions),
        function_prior=functions,
        pipeline_prior=p.JointInnovationPipelinePrior(
            channel_importance=False,
            l2_normalization=False,
            converter=p.JointNumericalConverterPrior(
                allow_activation=False, allow_kumaraswamy=False
            ),
            innovation=p.JointInnovationPrior(
                families=(family,),
                factors=("gaussian",),
                truncation=(None,),
                strengths=(0.5,),
                powers=(1.0,),
                structural_scale=(0.2, 0.2),
            ),
        ),
    )


def sample(request: p.GeometryPipelineRequest) -> p.SampledGeometryGraph:
    return p.sample_geometry_graph(
        request,
        named_rng=np.random.default_rng(11),
        raw_rng=np.random.default_rng(12),
        calibration_rng=np.random.default_rng(13),
        fit_rng=np.random.default_rng(14),
    )


@pytest.mark.parametrize("family", ["gaussian", "uniform", "student_t", "mixture", "ball"])
def test_sampled_complete_innovation_archive(family: str) -> None:
    graph = sample(request_for(family))
    assert p.recipe_from_dict(json.loads(json.dumps(asdict(graph)))) == graph
    for node in graph.runtime.nodes:
        assert node.exogenous.geometry == ("ball" if family == "ball" else "independent")
        assert node.exogenous.inputs.distribution == ("gaussian" if family == "ball" else family)
        assert len(node.exogenous.factor) == node.function.output_width
        assert node.noise_response.strength == 0.5
        assert node.noise_scale == (0.2,) * node.function.output_width
    task = p.generate_task(graph, seed=17, task_id=19)
    assert p.explain_generation(task, [15, 0, 15])["rows"] == [15, 0, 15]


@pytest.mark.parametrize("mode", ["local", "global", "meta"])
@pytest.mark.parametrize("scope", ["node", "channel"])
def test_bounded_heavy_tail_parameters_and_shared_scope(mode: str, scope: str) -> None:
    request = request_for("student_t")
    prior = request.pipeline_prior
    assert isinstance(prior, p.JointInnovationPipelinePrior)
    law = replace(
        prior.innovation,
        factors=("identity",),
        truncation=(2.0,),
        degrees=(1.0,),
        scale_scope=scope,
        structural_scale=(0.1, 0.9),
    )
    graph = sample(replace(request, mode=mode, pipeline_prior=replace(prior, innovation=law)))
    assert graph.replay() == graph
    for node in graph.runtime.nodes:
        assert node.exogenous.inputs.bound == 2.0
        assert node.exogenous.inputs.df == 1.0
        assert node.exogenous.factor == ()
        if scope == "node":
            assert len(set(node.noise_scale)) == 1
    assert any(r.shape[0] >= 256 for r in graph.graph.raw if r.distribution == "student_t")


def test_unsupported_heavy_tail_normalization_is_atomic() -> None:
    request = request_for("student_t")
    prior = request.pipeline_prior
    assert isinstance(prior, p.JointInnovationPipelinePrior)
    request = replace(
        request,
        pipeline_prior=replace(
            prior,
            l2_normalization=True,
            innovation=replace(prior.innovation, degrees=(1.0,), strengths=(0.0,)),
        ),
    )
    streams = {
        name: np.random.default_rng(i)
        for i, name in enumerate(("named_rng", "raw_rng", "calibration_rng", "fit_rng"))
    }
    original = json.dumps([r.bit_generator.state for r in streams.values()])
    with pytest.raises(ValueError, match="moment|variance|normaliz"):
        p.sample_geometry_graph(request, **streams)
    assert json.dumps([r.bit_generator.state for r in streams.values()]) == original


def test_v2_rejects_legacy_sigma_and_archive_mutations() -> None:
    with pytest.raises(ValueError, match="legacy Gaussian"):
        p.JointInnovationPipelinePrior(gaussian_noise=True)
    graph = sample(request_for("uniform"))
    raw = json.loads(json.dumps(asdict(graph)))
    raw["pipelines"][0]["noise"]["scales"][0] *= 2
    with pytest.raises(ValueError, match="replay"):
        p.recipe_from_dict(raw)
    assert isinstance(graph.request, p.GeometryPipelineRequest)
    for field in ("families", "truncation", "offset", "structural_scale"):
        raw = asdict(graph.request.pipeline_prior)
        raw["innovation"][field] = None
        with pytest.raises(ValueError, match="array|malformed|frozen|catalog"):
            p.pipeline_prior_from_dict(raw)


@pytest.mark.parametrize(
    "topology", [p.CauchyGraphPrior(nodes=3, offset=0.0), p.MotifGraphPrior(depth=1)]
)
def test_parent_composed_innovations_share_complete_laws(topology: object) -> None:
    request = request_for("mixture")
    assert isinstance(topology, (p.CauchyGraphPrior, p.MotifGraphPrior))
    controls = {
        field.name: getattr(request, field.name)
        for field in fields(p.GeometryGraphRequest)
        if field.name != "version"
    }
    controls["topology"] = topology
    composed = p.GeometryParentRequest(
        **controls,
        pipeline_prior=request.pipeline_prior,
        composition=p.ParentCompositionPrior(
            source="extension", mode="aggregate", aggregation="sum"
        ),
    )
    graph = p.sample_geometry_graph(
        composed,
        **{
            name: np.random.default_rng(i + 10)
            for i, name in enumerate(("named_rng", "raw_rng", "calibration_rng", "fit_rng"))
        },
    )
    assert graph.replay() == graph
    assert all(node.exogenous.inputs.distribution == "mixture" for node in graph.runtime.nodes)
    assert p.generate_task(graph, seed=12, task_id=13).inputs.shape == (16, 2)


@pytest.mark.parametrize("engine", [np.random.PCG64, np.random.Philox])
def test_extended_innovation_budget_failures_preserve_four_streams(
    engine: type[np.random.BitGenerator],
) -> None:
    request = request_for("uniform")
    for raw_limits in (p.RawDrawLimits(draws=1), p.RawDrawLimits(numbers=1)):
        streams = {
            name: np.random.Generator(engine(i))
            for i, name in enumerate(("named_rng", "raw_rng", "calibration_rng", "fit_rng"))
        }
        before = [pickle.dumps(r.bit_generator.state) for r in streams.values()]
        with pytest.raises(ValueError, match="limit|budget"):
            p.sample_geometry_graph(replace(request, raw_limits=raw_limits), **streams)
        assert [pickle.dumps(r.bit_generator.state) for r in streams.values()] == before
