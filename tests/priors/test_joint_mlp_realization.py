"""MLP fitting follows actual preceding outputs and retains every fitted stage."""

import copy
import json
import pickle
from dataclasses import asdict

import numpy as np
import pytest
import torch

from r0.modules import priors


def test_joint_mlp_relu_wrapper_fits_linear_outputs_and_preserves_raw_stage_order() -> None:
    context = priors.JointSamplingContext(np.random.default_rng(645), mode="local")
    plan = priors.sample_joint_function(
        context,
        "mlp",
        input_width=2,
        output_width=2,
        calibration_rows=4,
        prior=priors.JointFunctionPrior(
            functions="mlp", matrix=priors.JointMatrixPrior(families=("gaussian",))
        ),
    )
    assert dict(plan.parameters) == {
        "width": 2,
        "layers": 2,
        "start_activation": False,
        "end_activation": False,
    }
    assert plan.activations[0].activation.kind == "relu"
    sample = np.array([[0.0, 1.0], [2.0, -1.0], [-2.0, 4.0], [3.0, 2.0]])
    raw_oracle, fit_oracle = np.random.default_rng(101), np.random.default_rng(102)
    first_draw = raw_oracle.normal(size=(1, 2, 2))
    raw_oracle.normal(size=(1, 2, 2))
    shift_rows = fit_oracle.integers(4, size=2)
    second_draw = raw_oracle.normal(size=(1, 2, 2))
    raw_oracle.normal(size=(1, 2, 2))
    raw, fit = np.random.default_rng(101), np.random.default_rng(102)
    result = priors.materialize_joint_function(
        plan,
        context.freeze(),
        rng=raw,
        fit_rng=fit,
        calibration=sample,
        calibration_id="mlp-relu-fit",
    )
    assert isinstance(result, priors.JointMLPFunctionRealization)
    for matrix, expected in zip(result.matrices, (first_draw, second_draw), strict=True):
        source = matrix.matrices[0].source
        assert isinstance(source, priors.GaussianWeights)
        np.testing.assert_array_equal(source.draws, expected[0])
    assert raw.bit_generator.state == raw_oracle.bit_generator.state
    assert fit.bit_generator.state == fit_oracle.bit_generator.state
    first, second = (np.asarray(matrix.materialize()[0]) for matrix in result.matrices)
    hidden = sample @ first.T
    mean, inverse = hidden.mean(axis=0), 1 / (hidden.std(axis=0) + 1e-4)
    standardized = (hidden - mean) * inverse
    shift = -standardized[shift_rows, np.arange(2)]
    scale = plan.activations[0].scale
    assert scale is not None
    activated = np.maximum(0, scale * (standardized + shift))
    output_mean, output_inverse = activated.mean(axis=0), 1 / (activated.std(axis=0) + 1e-4)
    expected_fit = ((activated - output_mean) * output_inverse) @ second.T
    np.testing.assert_allclose(result.activation_inputs[0].sample, hidden, rtol=1e-13, atol=1e-15)
    assert not np.allclose(hidden, sample)
    np.testing.assert_array_equal(result.function.activations[0].shift_rows, shift_rows)
    np.testing.assert_allclose(
        result.function.activations[0].input_mean, mean, rtol=1e-13, atol=1e-15
    )
    np.testing.assert_allclose(result.outputs, expected_fit, rtol=1e-12, atol=1e-14)
    query = np.array([[1.0, -2.0], [4.0, 0.5]])
    transformed = np.maximum(0, scale * ((query @ first.T - mean) * inverse + shift))
    expected_query = ((transformed - output_mean) * output_inverse) @ second.T
    actual = priors.evaluate_frozen_function(result.function, torch.tensor(query))
    np.testing.assert_allclose(actual.numpy(), expected_query, rtol=1e-12, atol=1e-14)
    result.validate(context.freeze())


def test_joint_mlp_json_reconstructs_fits_and_rejects_changed_stage_inputs_or_constants() -> None:
    context = priors.JointSamplingContext(np.random.default_rng(645), mode="local")
    plan = priors.sample_joint_function(
        context,
        "mlp",
        input_width=2,
        output_width=2,
        calibration_rows=4,
        prior=priors.JointFunctionPrior(
            functions="mlp", matrix=priors.JointMatrixPrior(families=("gaussian",))
        ),
    )
    result = priors.materialize_joint_function(
        plan,
        context.freeze(),
        rng=np.random.default_rng(101),
        fit_rng=np.random.default_rng(102),
        calibration=np.array([[0.0, 1.0], [2.0, -1.0], [-2.0, 4.0], [3.0, 2.0]]),
        calibration_id="mlp-json",
    )
    payload = json.loads(json.dumps(asdict(result)))
    assert (
        priors.joint_function_realization_from_dict(payload, decisions=context.freeze()) == result
    )
    changed = copy.deepcopy(payload)
    changed["activation_inputs"][0]["sample"][0][0] += 0.25
    with pytest.raises(ValueError, match="digest"):
        priors.joint_function_realization_from_dict(changed, decisions=context.freeze())
    changed = copy.deepcopy(payload)
    changed["activation_inputs"][0]["identity"] = "unrelated-stage"
    with pytest.raises(ValueError, match="preceding stage outputs"):
        priors.joint_function_realization_from_dict(changed, decisions=context.freeze())
    for path in (
        ("function", "activations", 0, "input_mean", 0),
        ("function", "linears", 0, "matrix", 0, 0),
        ("outputs", 0, 0),
    ):
        changed = copy.deepcopy(payload)
        node = changed
        for key in path[:-1]:
            node = node[key]
        node[path[-1]] += 0.25
        with pytest.raises(ValueError, match="construction"):
            priors.joint_function_realization_from_dict(changed, decisions=context.freeze())


@pytest.mark.parametrize(
    "family", ["gaussian", "importance", "factorized", "kernel", "transformed"]
)
@pytest.mark.parametrize(
    ("seed", "layers", "start", "end"),
    [
        (0, 1, False, True),
        (1, 1, True, False),
        (2, 3, False, True),
        (3, 3, True, False),
        (5, 2, False, False),
        (8, 1, True, True),
        (9, 2, True, True),
        (12, 1, False, False),
        (13, 3, False, False),
        (21, 2, False, True),
        (50, 2, True, False),
        (56, 3, True, True),
    ],
)
def test_joint_mlp_all_layer_activation_flags_and_matrix_families_replay(
    family: str,
    seed: int,
    layers: int,
    start: bool,
    end: bool,
) -> None:
    context = priors.JointSamplingContext(np.random.default_rng(seed), mode="local")
    plan = priors.sample_joint_function(
        context,
        "mlp",
        input_width=2,
        output_width=2,
        calibration_rows=4,
        prior=priors.JointFunctionPrior(
            functions="mlp", matrix=priors.JointMatrixPrior(families=(family,))
        ),
    )
    parameters = dict(plan.parameters)
    assert (parameters["layers"], parameters["start_activation"], parameters["end_activation"]) == (
        layers,
        start,
        end,
    )
    sample = np.array([[0.0, 1.0], [2.0, -1.0], [-2.0, 4.0], [3.0, 2.0]])
    result = priors.materialize_joint_function(
        plan,
        context.freeze(),
        rng=np.random.default_rng(101),
        fit_rng=np.random.default_rng(102) if plan.activations else None,
        calibration=sample,
        calibration_id="x" * 128,
    )
    assert isinstance(result, priors.JointMLPFunctionRealization)
    assert len(result.matrices) == layers
    assert len(result.activation_inputs) == layers - 1 + start + end
    actual = priors.evaluate_frozen_function(result.function, torch.tensor(sample))
    np.testing.assert_array_equal(actual.numpy(), result.outputs)
    payload = json.loads(json.dumps(asdict(result)))
    assert (
        priors.joint_function_realization_from_dict(payload, decisions=context.freeze()) == result
    )
    sample[:] = -123
    assert result.calibration.sample[0] == (0.0, 1.0)


def test_joint_mlp_activation_failure_rolls_back_matrix_and_fit_draws_then_recovers() -> None:
    context = priors.JointSamplingContext(np.random.default_rng(1541), mode="local")
    plan = priors.sample_joint_function(
        context,
        "mlp",
        input_width=1,
        output_width=1,
        calibration_rows=10_000,
        prior=priors.JointFunctionPrior(
            functions="mlp", matrix=priors.JointMatrixPrior(families=("gaussian",))
        ),
    )
    assert len(plan.matrices) == 1
    assert len(plan.activations) == 1
    assert dict(plan.parameters)["end_activation"] is True
    assert plan.activations[0].activation.kind == "exp"
    sample = np.zeros((10_000, 1), dtype=np.float64)
    sample[0, 0] = -100
    raw, fit = np.random.default_rng(101), np.random.default_rng(102)
    before = pickle.dumps((raw.bit_generator.state, fit.bit_generator.state))
    decisions = context.freeze()
    # The finite first linear output is standardized; the rare positive extreme
    # then overflows exp after the activation has selected its shift row.
    with pytest.raises(ValueError, match="nonfinite"):
        priors.materialize_joint_function(
            plan,
            decisions,
            rng=raw,
            fit_rng=fit,
            calibration=sample,
            calibration_id="failed-mlp-fit",
        )
    assert pickle.dumps((raw.bit_generator.state, fit.bit_generator.state)) == before
    assert context.freeze() == decisions
    valid = np.linspace(-1.0, 1.0, 10_000, dtype=np.float64)[:, None]
    recovered = priors.materialize_joint_function(
        plan,
        decisions,
        rng=raw,
        fit_rng=fit,
        calibration=valid,
        calibration_id="recovered-mlp-fit",
    )
    fresh = priors.materialize_joint_function(
        plan,
        decisions,
        rng=np.random.default_rng(101),
        fit_rng=np.random.default_rng(102),
        calibration=valid,
        calibration_id="recovered-mlp-fit",
    )
    assert recovered == fresh


def test_joint_mlp_admits_retained_stage_samples_before_copying_or_drawing() -> None:
    context = priors.JointSamplingContext(np.random.default_rng(645), mode="local")
    plan = priors.sample_joint_function(
        context,
        "mlp",
        input_width=2,
        output_width=2,
        calibration_rows=1_000_000,
        prior=priors.JointFunctionPrior(
            functions="mlp", matrix=priors.JointMatrixPrior(families=("gaussian",))
        ),
    )
    raw, fit = np.random.default_rng(101), np.random.default_rng(102)
    before = pickle.dumps((raw.bit_generator.state, fit.bit_generator.state))
    sample = np.broadcast_to(np.array([[np.nan, np.nan]]), (1_000_000, 2))
    with pytest.raises(ValueError, match="limit"):
        priors.materialize_joint_function(
            plan,
            context.freeze(),
            rng=raw,
            fit_rng=fit,
            calibration=sample,
            calibration_id="oversized-mlp-fit",
        )
    assert pickle.dumps((raw.bit_generator.state, fit.bit_generator.state)) == before
