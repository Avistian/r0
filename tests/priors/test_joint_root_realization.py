"""Root bases retain independent calibration draws before the mandatory function."""

import json
import pickle
from dataclasses import asdict

import numpy as np
import pytest
import torch

from r0.modules import priors


@pytest.mark.parametrize("family", ["uniform", "ball", "gaussian", "covariance"])
def test_joint_root_realizes_base_then_function_with_separate_calibration_stream(
    family: str,
) -> None:
    context = priors.JointSamplingContext(np.random.default_rng(110), mode="local")
    plan = priors.sample_joint_root(
        context,
        "root",
        width=2,
        calibration_rows=4,
        prior=priors.JointRootPrior(
            families=(family,),
            function=priors.JointFunctionPrior(
                functions="lin",
                matrix=priors.JointMatrixPrior(families=("gaussian",), vector_family="uniform"),
            ),
        ),
    )
    raw_oracle, sample_oracle = np.random.default_rng(111), np.random.default_rng(112)
    factor = None
    weights = None
    if family == "covariance":
        factor = raw_oracle.normal(size=(2, 2))
        weights = raw_oracle.random(2)
    base_family = plan.covariance_base if family == "covariance" else family
    draws = (
        sample_oracle.random((4, 2))
        if base_family == "uniform"
        else sample_oracle.normal(size=(4, 2))
    )
    radii = sample_oracle.random(4) if family == "ball" else np.array([], dtype=np.float64)
    expected = -1 + 2 * draws if base_family == "uniform" else draws
    if family == "ball":
        expected = draws / np.linalg.norm(draws, axis=1)[:, None] * np.sqrt(radii[:, None])
    if factor is not None:
        assert weights is not None
        expected = expected @ (factor * weights[None, :]).T
    gaussian = raw_oracle.normal(size=(1, 2, 2))
    raw_oracle.normal(size=(1, 2, 2))
    raw, sample, fit = (
        np.random.default_rng(111),
        np.random.default_rng(112),
        np.random.default_rng(113),
    )
    fit_before = fit.bit_generator.state
    result = priors.materialize_joint_root(
        plan,
        context.freeze(),
        rng=raw,
        calibration_rng=sample,
        fit_rng=fit,
        calibration_id="independent-root-fit",
    )
    assert isinstance(result, priors.JointRootRealization)
    np.testing.assert_array_equal(result.calibration.draws, draws)
    np.testing.assert_array_equal(result.calibration.radial_uniform, radii)
    np.testing.assert_allclose(result.calibration.values, expected, rtol=1e-14, atol=1e-15)
    if factor is not None:
        assert weights is not None
        assert result.calibration.base.factor is not None
        np.testing.assert_array_equal(result.calibration.base.gaussian_factor, factor)
        np.testing.assert_array_equal(result.calibration.base.factor, factor * weights[None, :])
        assert not np.allclose(result.calibration.base.factor, weights[:, None] * factor)
    assert isinstance(result.function, priors.JointMatrixFunctionRealization)
    source = result.function.matrix.matrices[0].source
    assert isinstance(source, priors.GaussianWeights)
    np.testing.assert_array_equal(source.draws, gaussian[0])
    matrix = np.asarray(result.function.matrix.materialize()[0])
    np.testing.assert_allclose(result.outputs, expected @ matrix.T, rtol=1e-13, atol=1e-15)
    assert not np.allclose(result.outputs, expected)
    assert raw.bit_generator.state == raw_oracle.bit_generator.state
    assert sample.bit_generator.state == sample_oracle.bit_generator.state
    assert fit.bit_generator.state == fit_before
    result.validate(context.freeze())


@pytest.mark.parametrize("family", ["uniform", "ball", "gaussian", "covariance"])
@pytest.mark.parametrize("function", ["lin", "quad", "gp", "tree", "disc", "em", "mlp", "prod"])
def test_joint_root_all_bases_and_functions_use_retained_base_calibration(
    family: str, function: str
) -> None:
    context = priors.JointSamplingContext(np.random.default_rng(114), mode="local")
    plan = priors.sample_joint_root(
        context,
        "root",
        width=2,
        calibration_rows=4,
        prior=priors.JointRootPrior(
            families=(family,),
            function=priors.JointFunctionPrior(
                functions=function,
                frequencies=4,
                matrix=priors.JointMatrixPrior(families=("gaussian",)),
            ),
        ),
    )
    result = priors.materialize_joint_root(
        plan,
        context.freeze(),
        rng=np.random.default_rng(115),
        calibration_rng=np.random.default_rng(116),
        fit_rng=np.random.default_rng(117),
        calibration_id="sampled-root-fit",
    )
    restored = priors.joint_root_realization_from_dict(
        json.loads(json.dumps(asdict(result))), decisions=context.freeze()
    )
    assert restored == result
    actual = priors.evaluate_frozen_function(
        result.function.function,
        torch.tensor(result.calibration.values, dtype=torch.float64),
    )
    np.testing.assert_array_equal(result.outputs, actual.numpy())
    if isinstance(
        result.function,
        (
            priors.JointTreeFunctionRealization,
            priors.JointDiscretizationFunctionRealization,
            priors.JointEMFunctionRealization,
            priors.JointMLPFunctionRealization,
            priors.JointProductFunctionRealization,
        ),
    ):
        fitted = result.function.calibration
        if fitted is not None:
            assert fitted.sample == result.calibration.values
            assert fitted.identity == "sampled-root-fit"
    result.validate(context.freeze())


@pytest.mark.parametrize(("left", "right"), [(0, 1), (0, 2), (1, 2)])
def test_joint_root_rejects_aliased_streams_without_consuming_any_draws(
    left: int, right: int
) -> None:
    context = priors.JointSamplingContext(np.random.default_rng(110), mode="local")
    plan = priors.sample_joint_root(context, "root", width=2, calibration_rows=4)
    streams = [np.random.default_rng(seed) for seed in (111, 112, 113)]
    streams[right] = np.random.Generator(streams[left].bit_generator)
    before = pickle.dumps([stream.bit_generator.state for stream in streams])
    with pytest.raises(ValueError, match="distinct injected RNGs"):
        priors.materialize_joint_root(
            plan,
            context.freeze(),
            rng=streams[0],
            calibration_rng=streams[1],
            fit_rng=streams[2],
            calibration_id="aliased-root",
        )
    assert pickle.dumps([stream.bit_generator.state for stream in streams]) == before


def test_joint_root_admission_precedes_calibration_draws() -> None:
    context = priors.JointSamplingContext(np.random.default_rng(110), mode="local")
    plan = priors.sample_joint_root(
        context,
        "root",
        width=2,
        calibration_rows=10**9,
        prior=priors.JointRootPrior(
            families=("uniform",), function=priors.JointFunctionPrior(functions="lin")
        ),
    )
    raw, sample, fit = (np.random.default_rng(seed) for seed in (111, 112, 113))
    before = pickle.dumps(
        (raw.bit_generator.state, sample.bit_generator.state, fit.bit_generator.state)
    )
    with pytest.raises(ValueError, match="limit"):
        priors.materialize_joint_root(
            plan,
            context.freeze(),
            rng=raw,
            calibration_rng=sample,
            fit_rng=fit,
            calibration_id="oversized-root",
        )
    assert (
        pickle.dumps((raw.bit_generator.state, sample.bit_generator.state, fit.bit_generator.state))
        == before
    )
