"""Sampled node pipelines survive complete construction and product replay."""

import json
from dataclasses import asdict, replace
from typing import cast

import numpy as np
import pytest

from r0.modules import priors
from r0.modules.episodes import TaskType


def _request(variant: str = "readout-only") -> priors.GeometryPipelineRequest:
    functions = priors.JointFunctionPrior(
        functions="lin", matrix=priors.JointMatrixPrior(families=("gaussian",))
    )
    return priors.GeometryPipelineRequest(
        topology=priors.AttachmentGraphPrior(nodes=3),
        task=priors.LinearRecipe(
            rows=16, context_rows=8, features=2, signal_features=2, active_features=2
        ),
        subsample=False,
        width_prior=priors.JointWidthPrior(low=1, high=3),
        calibration_rows=6,
        root_prior=priors.JointRootPrior(families=("gaussian",), function=functions),
        function_prior=functions,
        pipeline_prior=priors.JointPipelinePrior(
            gaussian_noise=True,
            converter=priors.JointNumericalConverterPrior(variant=variant),
        ),
    )


def _sample(request: priors.GeometryGraphRequest, seed: int = 100) -> priors.SampledGeometryGraph:
    return priors.sample_geometry_graph(
        request,
        named_rng=np.random.default_rng(seed),
        raw_rng=np.random.default_rng(seed + 1),
        calibration_rng=np.random.default_rng(seed + 2),
        fit_rng=np.random.default_rng(seed + 3),
    )


@pytest.mark.parametrize("variant", ["readout-only", "historical-tuple", "historical-graph"])
def test_pipeline_archive_generate_inspection_resplit_and_pairs(variant: str) -> None:
    result = _sample(_request(variant))
    assert isinstance(result, priors.SampledPipelineGeometry)
    assert result.replay() == result
    archived = json.loads(json.dumps(asdict(result)))
    assert priors.recipe_from_dict(archived) == result
    task = priors.generate_task(result, seed=17, task_id=2)
    assert priors.explain_generation(task, [7, 0, 7])["rows"] == [7, 0, 7]
    resplit = result.with_task(replace(result.task, context_rows=4, split=1))
    other = priors.generate_task(resplit, seed=17, task_id=2)
    assert task.source_task_id == other.source_task_id
    np.testing.assert_array_equal(task.inputs.numpy(), other.inputs.numpy())
    paired = priors.paired_recipe(result, intervention="noise", value=0.2)
    assert isinstance(paired, priors.SampledPipelineGeometry)
    assert paired.pipelines == result.pipelines
    for fitted in result.pipelines:
        values = np.asarray(fitted.calibration.sample)
        standard = fitted.pipeline.stages[0]
        assert isinstance(standard, priors.CalibratedStandardization)
        np.testing.assert_allclose(standard.mean, values.mean(axis=0), rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(
            standard.inverse_std, 1 / (values.std(axis=0) + 1e-4), rtol=1e-12
        )
        weights = np.asarray(fitted.pipeline.channel_importance)
        assert (weights >= 0).all()
        assert np.mean(weights**2) == pytest.approx(1)
        transformed = (values - standard.mean) * standard.inverse_std * weights
        assert fitted.pipeline.l2 is not None
        assert fitted.pipeline.l2.factor == pytest.approx(
            1 / (np.sqrt(np.mean(np.sum(transformed**2, axis=1))) + 1e-8)
        )


@pytest.mark.parametrize(
    "engine",
    [np.random.PCG64, np.random.PCG64DXSM, np.random.Philox, np.random.SFC64, np.random.MT19937],
)
def test_pipeline_advanced_engines_and_late_combined_budget_rollback(
    engine: type[np.random.BitGenerator],
) -> None:
    streams = [np.random.Generator(engine(300 + i)) for i in range(4)]
    for stream in streams:
        stream.random(7)
    result = priors.sample_geometry_graph(
        _request(),
        named_rng=streams[0],
        raw_rng=streams[1],
        calibration_rng=streams[2],
        fit_rng=streams[3],
    )
    assert result.replay() == result
    # Reserve enough for most mechanism/pipeline work, but not its last draw.
    total = len(result.topology.raw) + len(result.observations.raw) + len(result.graph.raw)
    request = replace(_request(), raw_limits=priors.RawDrawLimits(draws=total - 1))
    streams = [snapshot.restore() for snapshot in result.initial]
    with pytest.raises(ValueError, match="raw.*limit"):
        priors.sample_geometry_graph(
            request,
            named_rng=streams[0],
            raw_rng=streams[1],
            calibration_rng=streams[2],
            fit_rng=streams[3],
        )
    for stream, before in zip(streams, result.initial, strict=True):
        np.testing.assert_array_equal(stream.random(8), before.restore().random(8))


@pytest.mark.parametrize(
    "tamper",
    [
        "fit",
        "sigma",
        "variant",
        "missing-owner",
        "foreign-owner",
        "request-version",
        "missing-pipelines",
        "extra-field",
    ],
)
def test_pipeline_archive_rejects_changed_fits_bindings_and_versions(tamper: str) -> None:
    result = _sample(_request())
    archive = json.loads(json.dumps(asdict(result)))
    if tamper == "fit":
        archive["pipelines"][1]["calibration"]["sample"][0][0] += 0.01
    elif tamper == "sigma":
        archive["pipelines"][0]["plan"]["sigma"] *= 0.5
    elif tamper == "variant":
        archive["pipelines"][0]["plan"]["prior"]["converter"]["variant"] = "historical-tuple"
    elif tamper == "missing-owner":
        archive["graph"]["registry"]["operations"].pop()
    elif tamper == "foreign-owner":
        archive["graph"]["registry"]["operations"][-1]["operation"] = ["foreign"]
    elif tamper == "request-version":
        archive["request"]["version"] = "geometry-graph-request-v1"
    elif tamper == "missing-pipelines":
        archive.pop("pipelines")
    else:
        archive["request"]["pipeline_prior"]["unknown"] = True
    with pytest.raises(ValueError, match="unknown fields|differs from complete replay"):
        priors.recipe_from_dict(archive)


def test_pipeline_calibration_uses_final_parent_states_and_independent_population() -> None:
    result = _sample(_request("historical-graph"))
    assert isinstance(result, priors.SampledPipelineGeometry)
    receiving = 2
    fitted = result.pipelines[receiving]
    streams = {
        role: snapshot.restore()
        for role, snapshot in zip(
            ("named", "raw", "calibration", "fit"), result.initial, strict=True
        )
    }
    for record in (
        *result.topology.raw,
        *result.observations.raw,
        *result.graph.raw[: fitted.raw_start],
    ):
        record.replay(streams[record.stream])
    calibration_before = streams["calibration"].bit_generator.state
    current = replace(result.runtime.nodes[receiving], pipeline=None, noise_scale=())
    nodes = (*result.runtime.nodes[:receiving], current)
    prefix = priors.FunctionalPrefix(nodes, result.request.calibration_rows, result.request.limits)
    expected = priors.sample_functional_prefix(
        prefix, rng=streams["calibration"], owner=("oracle",)
    )
    np.testing.assert_array_equal(expected.states[-1].numpy(), fitted.calibration.sample)
    # Counterfactual: ignoring final parent normalization/conversion/noise changes
    # the receiving fit, even when replaying exactly the same population stream.
    wrong_nodes = tuple(replace(n, pipeline=None, noise_scale=()) for n in nodes)
    streams["calibration"].bit_generator.state = calibration_before
    wrong = priors.sample_functional_prefix(
        priors.FunctionalPrefix(
            wrong_nodes, result.request.calibration_rows, result.request.limits
        ),
        rng=streams["calibration"],
        owner=("counterfactual",),
    )
    assert not np.allclose(wrong.states[-1].numpy(), fitted.calibration.sample)
    changed = result.with_task(replace(result.task, rows=24, context_rows=5))
    assert isinstance(changed, priors.SampledPipelineGeometry)
    assert changed.pipelines == result.pipelines
    assert changed.graph.raw == result.graph.raw


@pytest.mark.parametrize("family", ["polynomial", "uniform"])
def test_pipeline_target_restriction_and_optional_stages(family: str) -> None:
    request = _request()
    prior = replace(
        request.pipeline_prior,
        weight_family=family,
        l2_normalization=False,
        node_importance=False,
        gaussian_noise=False,
        converter=priors.JointNumericalConverterPrior(disallow_target_warping=True),
    )
    result = _sample(replace(request, pipeline_prior=prior))
    assert isinstance(result, priors.SampledPipelineGeometry)
    for fitted in result.pipelines:
        assert fitted.pipeline.l2 is None
        assert fitted.pipeline.node_importance == 1
        assert fitted.plan.sigma == 0
        for converter in fitted.plan.converters:
            if converter.feature_group == "y":
                assert converter.mode == "id"
    disabled = replace(prior, channel_importance=False)
    other = _sample(replace(request, pipeline_prior=disabled))
    assert isinstance(other, priors.SampledPipelineGeometry)
    assert all(
        p.pipeline.stages == () and p.pipeline.channel_importance == () and p.weights is None
        for p in other.pipelines
    )
    assert other.replay() == other


@pytest.mark.parametrize("family", ["tree", "mlp"])
def test_fitted_mechanisms_and_pipeline_populations_have_distinct_receipts(family: str) -> None:
    request = _request()
    functions = replace(request.function_prior, functions=family)
    result = _sample(
        replace(
            request,
            function_prior=functions,
            root_prior=replace(request.root_prior, function=functions),
        ),
        seed=114,
    )
    assert isinstance(result, priors.SampledPipelineGeometry)
    assert result.replay() == result
    assert len(result.graph.calibrations) == 2
    for receipt in result.graph.calibrations:
        fitted = result.pipelines[receipt.node]
        assert receipt.raw_stop <= fitted.raw_start
        assert receipt.identity != fitted.calibration.identity
    emitted = priors.generate_task(result, seed=2, task_id=4)
    assert priors.explain_generation(emitted, [0, 2])["rows"] == [0, 2]


@pytest.mark.parametrize(
    "topology",
    [
        priors.AttachmentGraphPrior(nodes=3, components=3),
        priors.MotifGraphPrior(depth=0, components=3),
    ],
)
def test_pipeline_empty_topology_prefix_and_singleton_calibration(
    topology: priors.AttachmentGraphPrior | priors.MotifGraphPrior,
) -> None:
    result = _sample(replace(_request(), topology=topology, calibration_rows=1))
    assert isinstance(result, priors.SampledPipelineGeometry)
    assert result.topology.raw == ()
    assert result.replay() == result
    for fitted in result.pipelines:
        standard = fitted.pipeline.stages[0]
        assert isinstance(standard, priors.CalibratedStandardization)
        assert standard.inverse_std == (10000.0,) * fitted.plan.width
        assert fitted.pipeline.l2 is not None
        assert fitted.pipeline.l2.factor == 1e8


@pytest.mark.parametrize("resource", ["parameters", "working_bytes"])
def test_pipeline_aggregate_retained_fit_admission_rolls_back_all_streams(resource: str) -> None:
    result = _sample(_request())
    limits = replace(
        result.request.limits, **{resource: 100_000 if resource == "parameters" else 40 * 1024**2}
    )
    # Large independent fit populations are charged before pipeline ndarray work.
    request = replace(_request(), calibration_rows=2048, limits=limits)
    streams = [snapshot.restore() for snapshot in result.initial]
    with pytest.raises(ValueError, match="aggregate.*limit"):
        priors.sample_geometry_graph(
            request,
            named_rng=streams[0],
            raw_rng=streams[1],
            calibration_rng=streams[2],
            fit_rng=streams[3],
        )
    for stream, before in zip(streams, result.initial, strict=True):
        np.testing.assert_array_equal(stream.random(5), before.restore().random(5))


@pytest.mark.parametrize("mode", ["local", "meta", "global"])
def test_pipeline_sampling_policies_and_catalog_restrictions(mode: str) -> None:
    request = _request()
    request = replace(
        request,
        mode=mode,
        pipeline_prior=replace(
            request.pipeline_prior,
            converter=priors.JointNumericalConverterPrior(
                allow_kumaraswamy=False, allow_activation=False
            ),
        ),
    )
    result = _sample(request)
    assert isinstance(result, priors.SampledPipelineGeometry)
    assert result.replay() == result
    assert all(c.mode == "id" for fitted in result.pipelines for c in fitted.plan.converters)
    # Matrix and pipeline requests are one shared registry, with no cloned outer sampler.
    names = [p.request.name for p in result.graph.registry.decisions.numerical.preferences]
    assert names.count("node_importance") == 1
    assert names.count("gaussian_noise_sigma") == 1


def test_pipeline_readout_capture_precedes_noise_and_node_importance() -> None:
    results: dict[str, priors.SampledPipelineGeometry] = {}
    evaluations: dict[str, priors.PrefixEvaluation] = {}
    for variant in ("readout-only", "historical-tuple", "historical-graph"):
        result = _sample(replace(_request(variant), topology=priors.AttachmentGraphPrior(nodes=1)))
        assert isinstance(result, priors.SampledPipelineGeometry)
        results[variant] = result
        evaluations[variant] = priors.sample_functional_prefix(
            priors.FunctionalPrefix(result.runtime.nodes, 8, result.request.limits),
            rng=np.random.default_rng(39),
            owner=("readout-oracle",),
        )
    graph = results["historical-graph"]
    channels = [c.channel for c in graph.pipelines[0].plan.converters if c.mode != "id"]
    assert channels  # This retained seed exercises a nonidentity readout.
    np.testing.assert_array_equal(
        evaluations["historical-tuple"].states[0].numpy(),
        evaluations["historical-graph"].states[0].numpy(),
    )
    np.testing.assert_array_equal(
        evaluations["readout-only"].readouts[0][:, channels].numpy(),
        evaluations["historical-graph"].readouts[0][:, channels].numpy(),
    )
    assert not np.allclose(
        evaluations["historical-tuple"].readouts[0][:, channels].numpy(),
        evaluations["historical-graph"].readouts[0][:, channels].numpy(),
    )
    node = graph.runtime.nodes[0]
    assert node.pipeline is not None
    before_noise = replace(node, noise_scale=(), pipeline=replace(node.pipeline, node_importance=1))
    clean = priors.sample_functional_prefix(
        priors.FunctionalPrefix((before_noise,), 8, graph.request.limits),
        rng=np.random.default_rng(39),
        owner=("clean-oracle",),
    )
    np.testing.assert_array_equal(
        clean.readouts[0][:, channels].numpy(),
        evaluations["historical-graph"].readouts[0][:, channels].numpy(),
    )
    assert not np.allclose(
        clean.states[0].numpy(), evaluations["historical-graph"].states[0].numpy()
    )


@pytest.mark.parametrize(("task_type", "classes"), [("binary", 2), ("multiclass", 3)])
def test_pipeline_label_adapters_preserve_sampled_mechanisms(task_type: str, classes: int) -> None:
    request = _request()
    request = replace(
        request, task=replace(request.task, task_type=cast(TaskType, task_type), classes=classes)
    )
    result = _sample(request)
    assert isinstance(result, priors.SampledPipelineGeometry)
    assert priors.recipe_from_dict(json.loads(json.dumps(asdict(result)))) == result
    task = priors.generate_task(result, seed=4, task_id=1)
    assert task.task_type == task_type
    assert priors.explain_generation(task, [3, 1])["rows"] == [3, 1]
    paired = priors.paired_recipe(result, intervention="separation", value=0.25)
    assert isinstance(paired, priors.SampledPipelineGeometry)
    assert paired.pipelines == result.pipelines
    assert paired.graph.raw == result.graph.raw
