"""Tree fits retain independent samples, selected coordinates and raw leaf draws."""

import copy
import json
import pickle
from dataclasses import asdict
from typing import Any, cast

import numpy as np
import pytest
import torch

from r0.modules import priors


def test_joint_tree_realization_binds_threshold_rows_and_leaf_draws_to_the_fit_sample() -> None:
    context = priors.JointSamplingContext(np.random.default_rng(70), mode="local")
    plan = priors.sample_joint_function(
        context,
        "tree",
        input_width=2,
        output_width=1,
        calibration_rows=2,
        prior=priors.JointFunctionPrior(functions="tree"),
    )
    parameters = dict(plan.parameters)
    trees, depth = int(parameters["trees"]), int(parameters["depth"])
    calibration = np.array([[0.0, 7.0], [1.0, 7.0]])
    fit_oracle = np.random.default_rng(71)
    probabilities = np.array([0.5, 1e-8]) / (0.5 + 1e-8)
    features = fit_oracle.choice(2, size=(trees, depth), p=probabilities)
    rows = fit_oracle.integers(2, size=(trees, depth))
    leaves = np.random.default_rng(72).normal(size=(trees, 2**depth, 1))
    result = priors.materialize_joint_function(
        plan,
        context.freeze(),
        rng=np.random.default_rng(72),
        fit_rng=np.random.default_rng(71),
        calibration=calibration,
        calibration_id="independent-tree-fit",
    )
    assert isinstance(result, priors.JointTreeFunctionRealization)
    np.testing.assert_array_equal(result.features, features)
    np.testing.assert_array_equal(result.threshold_rows, rows)
    np.testing.assert_array_equal(result.leaf_gaussian, leaves)
    assert result.feature_weights == (0.5, 1e-8)
    assert result.calibration.sample == ((0.0, 7.0), (1.0, 7.0))
    assert result.calibration.identity == "independent-tree-fit"
    np.testing.assert_array_equal(result.function.thresholds, calibration[rows, features])
    expected: list[list[float]] = []
    for x in (0.0, 1.0, 2.0):
        query = (x, 7.0)
        values: list[float] = []
        for tree in range(trees):
            index = sum(
                2**level
                for level in range(depth)
                if query[features[tree, level]]
                > calibration[rows[tree, level], features[tree, level]]
            )
            values.append(float(leaves[tree, index, 0]))
        expected.append([float(np.mean(values))])
    actual = priors.evaluate_frozen_function(
        result.function, torch.tensor([[0.0, 7.0], [1.0, 7.0], [2.0, 7.0]])
    )
    np.testing.assert_allclose(actual.numpy(), expected, rtol=1e-13, atol=1e-15)
    calibration[:] = -123
    assert result.calibration.sample == ((0.0, 7.0), (1.0, 7.0))
    result.validate(context.freeze())


def test_joint_tree_json_rejects_changed_fit_input_thresholds_and_rows() -> None:
    context = priors.JointSamplingContext(np.random.default_rng(70), mode="local")
    plan = priors.sample_joint_function(
        context,
        "tree",
        input_width=2,
        output_width=1,
        calibration_rows=2,
        prior=priors.JointFunctionPrior(functions="tree"),
    )
    result = priors.materialize_joint_function(
        plan,
        context.freeze(),
        rng=np.random.default_rng(72),
        fit_rng=np.random.default_rng(71),
        calibration=np.array([[0.0, 7.0], [1.0, 7.0]]),
        calibration_id="fit-tree-json",
    )
    payload = json.loads(json.dumps(asdict(result)))
    assert (
        priors.joint_function_realization_from_dict(payload, decisions=context.freeze()) == result
    )
    changed_input = copy.deepcopy(payload)
    changed_input["calibration"]["sample"][0][0] = 0.5
    with pytest.raises(ValueError, match="digest"):
        priors.joint_function_realization_from_dict(changed_input, decisions=context.freeze())
    changed_equation = copy.deepcopy(payload)
    changed_equation["function"]["thresholds"][0][0] += 0.5
    with pytest.raises(ValueError, match="construction"):
        priors.joint_function_realization_from_dict(changed_equation, decisions=context.freeze())
    payload["threshold_rows"][0][0] = 1 - payload["threshold_rows"][0][0]
    with pytest.raises(ValueError, match="construction"):
        priors.joint_function_realization_from_dict(payload, decisions=context.freeze())


def test_joint_tree_json_rejects_boolean_feature_mass_even_when_deviation_is_one() -> None:
    context = priors.JointSamplingContext(np.random.default_rng(70), mode="local")
    plan = priors.sample_joint_function(
        context,
        "tree",
        input_width=1,
        output_width=1,
        calibration_rows=2,
        prior=priors.JointFunctionPrior(functions="tree"),
    )
    result = priors.materialize_joint_function(
        plan,
        context.freeze(),
        rng=np.random.default_rng(72),
        fit_rng=np.random.default_rng(71),
        calibration=np.array([[-1.0], [1.0]]),
        calibration_id="unit-deviation",
    )
    payload = json.loads(json.dumps(asdict(result)))
    assert payload["feature_weights"] == [1.0]
    payload["feature_weights"] = [True]
    with pytest.raises(ValueError, match="feature masses"):
        priors.joint_function_realization_from_dict(payload, decisions=context.freeze())


@pytest.mark.parametrize(
    "case",
    [
        "missing-fit-rng",
        "shared-generator",
        "shared-bit-generator",
        "wrong-shape",
        "integer-sample",
        "nonfinite-sample",
        "empty-identity",
        "parameter-limit",
        "memory-limit",
    ],
)
def test_joint_tree_rejected_fit_preserves_both_rngs_and_named_decisions(case: str) -> None:
    context = priors.JointSamplingContext(np.random.default_rng(70), mode="local")
    plan = priors.sample_joint_function(
        context,
        "tree",
        input_width=2,
        output_width=1,
        calibration_rows=2,
        prior=priors.JointFunctionPrior(functions="tree"),
    )
    decisions = context.freeze()
    raw = np.random.default_rng(72)
    fit = np.random.default_rng(71)
    fit_argument = fit
    sample = np.array([[0.0, 7.0], [1.0, 7.0]])
    identity = "transactional-tree-fit"
    limits = priors.CausalLimits()
    if case == "missing-fit-rng":
        fit_argument = None
    elif case == "shared-generator":
        fit_argument = raw
    elif case == "shared-bit-generator":
        fit_argument = np.random.Generator(raw.bit_generator)
    elif case == "wrong-shape":
        sample = sample[:1]
    elif case == "integer-sample":
        sample = cast(Any, sample.astype(np.int64))
    elif case == "nonfinite-sample":
        sample[0, 0] = np.nan
    elif case == "empty-identity":
        identity = ""
    elif case == "parameter-limit":
        limits = priors.CausalLimits(parameters=1)
    else:
        limits = priors.CausalLimits(working_bytes=1)
    raw_before, fit_before = (
        pickle.dumps(raw.bit_generator.state),
        pickle.dumps(fit.bit_generator.state),
    )
    with pytest.raises(ValueError, match="separate injected fit RNG|calibration|limit"):
        priors.materialize_joint_function(
            plan,
            decisions,
            rng=raw,
            fit_rng=fit_argument,
            calibration=sample,
            calibration_id=identity,
            limits=limits,
        )
    assert pickle.dumps(raw.bit_generator.state) == raw_before
    assert pickle.dumps(fit.bit_generator.state) == fit_before
    assert context.freeze() == decisions


def test_joint_tree_large_calibration_is_admitted_before_sample_validation() -> None:
    context = priors.JointSamplingContext(np.random.default_rng(70), mode="local")
    plan = priors.sample_joint_function(
        context,
        "tree",
        input_width=2,
        output_width=1,
        calibration_rows=1_000_000,
        prior=priors.JointFunctionPrior(functions="tree"),
    )
    raw, fit = np.random.default_rng(72), np.random.default_rng(71)
    before = pickle.dumps((raw.bit_generator.state, fit.bit_generator.state))
    # A read-only broadcast view carries oversized geometry without allocating it.
    # NaNs distinguish rejection by admission from later finite-input validation.
    sample = np.broadcast_to(np.array([[np.nan, np.nan]]), (1_000_000, 2))
    with pytest.raises(ValueError, match="limit"):
        priors.materialize_joint_function(
            plan,
            context.freeze(),
            rng=raw,
            fit_rng=fit,
            calibration=sample,
            calibration_id="oversized-tree-fit",
        )
    assert pickle.dumps((raw.bit_generator.state, fit.bit_generator.state)) == before
