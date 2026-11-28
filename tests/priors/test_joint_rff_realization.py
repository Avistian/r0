"""Product and radial Fourier constructions retain their actual source draws."""

import copy
import json
from dataclasses import asdict

import numpy as np
import pytest
import torch

from r0.modules import priors


def test_joint_product_gp_retains_raw_uniform_spectrum_phases_and_gaussian_coefficients() -> None:
    context = priors.JointSamplingContext(np.random.default_rng(2), mode="global")
    plan = priors.sample_joint_function(
        context,
        "gp",
        input_width=3,
        output_width=2,
        calibration_rows=8,
        prior=priors.JointFunctionPrior(functions="gp", frequencies=4),
    )
    assert dict(plan.parameters)["product_kernel"] is True
    expected_rng = np.random.default_rng(60)
    uniform = expected_rng.random((3, 4))
    phase = expected_rng.random(4)
    gaussian = expected_rng.normal(size=(4, 2))
    result = priors.materialize_joint_function(
        plan, context.freeze(), rng=np.random.default_rng(60)
    )
    assert isinstance(result, priors.JointRFFFunctionRealization)
    np.testing.assert_array_equal(result.uniform, uniform)
    np.testing.assert_array_equal(result.phase_uniform, phase)
    np.testing.assert_array_equal(result.coefficient_gaussian, gaussian)
    assert result.directions is None
    assert result.factor is None
    assert result.weights is None
    a = dict(plan.parameters)["a"]
    frequency = np.clip(uniform, 1e-6, 1 - 1e-6) ** (1 / (1 - a)) - 1
    np.testing.assert_allclose(result.function.frequencies, frequency, rtol=1e-14)
    # At zero input, only the retained phase and output coefficients contribute.
    expected = np.cos(2 * np.pi * phase) @ gaussian / 2
    actual = priors.evaluate_frozen_function(result.function, torch.zeros((1, 3)))
    np.testing.assert_allclose(actual.numpy()[0], expected, rtol=1e-14, atol=1e-15)
    result.validate(context.freeze())


def test_joint_radial_gp_weights_factor_rows_and_keeps_raw_draw_order() -> None:
    context = priors.JointSamplingContext(np.random.default_rng(0), mode="global")
    plan = priors.sample_joint_function(
        context,
        "gp",
        input_width=3,
        output_width=2,
        calibration_rows=8,
        prior=priors.JointFunctionPrior(
            functions="gp",
            frequencies=4,
            matrix=priors.JointMatrixPrior(vector_family="uniform"),
        ),
    )
    parameters = dict(plan.parameters)
    assert parameters["product_kernel"] is False
    expected_rng = np.random.default_rng(61)
    uniform = expected_rng.random(4)
    directions = expected_rng.normal(size=(3, 4))
    factor = expected_rng.normal(size=(3, 3))
    weights = expected_rng.random((1, 3))[0]
    phase = expected_rng.random(4)
    coefficients = expected_rng.normal(size=(4, 2))
    result = priors.materialize_joint_function(
        plan, context.freeze(), rng=np.random.default_rng(61)
    )
    assert isinstance(result, priors.JointRFFFunctionRealization)
    np.testing.assert_array_equal(result.uniform, uniform[None, :])
    np.testing.assert_array_equal(result.directions, directions)
    np.testing.assert_array_equal(result.factor, factor)
    assert isinstance(result.weights, priors.UniformWeightVector)
    np.testing.assert_array_equal(result.weights.draws, weights)
    np.testing.assert_array_equal(result.phase_uniform, phase)
    np.testing.assert_array_equal(result.coefficient_gaussian, coefficients)
    radii = np.clip(uniform, 1e-6, 1 - 1e-6) ** (1 / (1 - parameters["a"])) - 1
    points = directions * radii / np.sqrt(np.sum(directions**2, axis=0))
    expected = parameters["scale"] * (weights[:, None] * factor) @ points
    np.testing.assert_allclose(result.function.frequencies, expected, rtol=1e-13, atol=1e-15)
    assert not np.allclose(expected, parameters["scale"] * (weights[None, :] * factor) @ points)
    result.validate(context.freeze())


def test_joint_gp_json_roundtrips_both_spectra_and_binds_the_retained_equation() -> None:
    for seed in (0, 2):
        for weight_family in ("uniform", "polynomial"):
            context = priors.JointSamplingContext(np.random.default_rng(seed), mode="global")
            plan = priors.sample_joint_function(
                context,
                "gp",
                input_width=3,
                output_width=2,
                calibration_rows=8,
                prior=priors.JointFunctionPrior(
                    functions="gp",
                    frequencies=4,
                    matrix=priors.JointMatrixPrior(vector_family=weight_family),
                ),
            )
            result = priors.materialize_joint_function(
                plan, context.freeze(), rng=np.random.default_rng(62)
            )
            payload = json.loads(json.dumps(asdict(result)))
            assert (
                priors.joint_function_realization_from_dict(payload, decisions=context.freeze())
                == result
            )
            payload["function"]["frequencies"][0][0] += 0.25
            with pytest.raises(ValueError, match="construction"):
                priors.joint_function_realization_from_dict(payload, decisions=context.freeze())


def test_joint_radial_gp_rejects_zero_directions_but_preserves_zero_factor_weights() -> None:
    context = priors.JointSamplingContext(np.random.default_rng(0), mode="global")
    plan = priors.sample_joint_function(
        context,
        "gp",
        input_width=3,
        output_width=2,
        calibration_rows=8,
        prior=priors.JointFunctionPrior(
            functions="gp",
            frequencies=4,
            matrix=priors.JointMatrixPrior(vector_family="uniform"),
        ),
    )
    result = priors.materialize_joint_function(
        plan, context.freeze(), rng=np.random.default_rng(63)
    )
    payload = json.loads(json.dumps(asdict(result)))
    zero_direction = copy.deepcopy(payload)
    for row in zero_direction["directions"]:
        row[0] = 0.0
    with pytest.raises(ValueError, match="zero Fourier direction"):
        priors.joint_function_realization_from_dict(zero_direction, decisions=context.freeze())
    payload["weights"]["draws"] = [0.0, 0.0, 0.0]
    payload["function"]["frequencies"] = [[0.0] * 4 for _ in range(3)]
    restored = priors.joint_function_realization_from_dict(payload, decisions=context.freeze())
    assert isinstance(restored, priors.JointRFFFunctionRealization)
    assert restored.function.frequencies == ((0.0,) * 4,) * 3


def test_joint_gp_admits_all_frequency_arrays_before_drawing_or_allocating_them() -> None:
    for seed in (0, 2):
        context = priors.JointSamplingContext(np.random.default_rng(seed), mode="global")
        plan = priors.sample_joint_function(
            context,
            "gp",
            input_width=3,
            output_width=2,
            calibration_rows=8,
            prior=priors.JointFunctionPrior(functions="gp", frequencies=10**9),
        )
        rng = np.random.default_rng(64)
        state = copy.deepcopy(rng.bit_generator.state)
        before = context.freeze()
        with pytest.raises(ValueError, match="limit"):
            priors.materialize_joint_function(plan, before, rng=rng)
        assert rng.bit_generator.state == state
        assert context.freeze() == before
