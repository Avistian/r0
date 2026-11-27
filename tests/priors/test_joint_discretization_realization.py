"""Discretization freezes selected fit rows and their source linear targets."""

import copy
import json
import pickle
from dataclasses import asdict

import numpy as np
import pytest
import torch

from r0.modules import priors


@pytest.mark.parametrize(("rows", "requested"), [(1, None), (2, None), (3, 2), (3, 9), (7, None)])
def test_joint_discretization_preserves_center_order_targets_and_single_row_rng(
    rows: int,
    requested: int | None,
) -> None:
    context = priors.JointSamplingContext(np.random.default_rng(80), mode="local")
    plan = priors.sample_joint_function(
        context,
        "discretization",
        input_width=2,
        output_width=2,
        calibration_rows=rows,
        prior=priors.JointFunctionPrior(
            functions="disc",
            discretization_centers=requested,
            matrix=priors.JointMatrixPrior(families=("gaussian",)),
        ),
    )
    sample = np.column_stack((2.0 * np.arange(rows), np.zeros(rows))).astype(np.float64)
    fit_oracle, raw_oracle = np.random.default_rng(81), np.random.default_rng(82)
    order = tuple(map(int, fit_oracle.permutation(rows))) if rows > 1 else ()
    selected = order[: plan.center_count] if rows > 1 else (0,)
    gaussian = raw_oracle.normal(size=(1, 2, 2))
    raw_oracle.normal(size=(1, 2, 2))  # Source normalization jitter follows the matrix draw.
    fit, raw = np.random.default_rng(81), np.random.default_rng(82)
    result = priors.materialize_joint_function(
        plan,
        context.freeze(),
        rng=raw,
        fit_rng=fit,
        calibration=sample,
        calibration_id="independent-discretization-fit",
    )
    assert isinstance(result, priors.JointDiscretizationFunctionRealization)
    assert result.permutation == order
    np.testing.assert_array_equal(result.function.centers, sample[list(selected)])
    matrix_source = result.matrix.matrices[0].source
    assert isinstance(matrix_source, priors.GaussianWeights)
    np.testing.assert_array_equal(matrix_source.draws, gaussian[0])
    assert pickle.dumps(fit.bit_generator.state) == pickle.dumps(fit_oracle.bit_generator.state)
    assert pickle.dumps(raw.bit_generator.state) == pickle.dumps(raw_oracle.bit_generator.state)
    matrix = result.matrix.materialize()[0]
    expected = [[2 * row * matrix[0][0], 2 * row * matrix[1][0]] for row in selected]
    np.testing.assert_array_equal(result.targets, expected)
    actual = priors.evaluate_frozen_function(
        result.function, torch.tensor(sample[list(selected)], dtype=torch.float64)
    )
    np.testing.assert_array_equal(actual.numpy(), expected)
    if len(selected) == 2:
        # The source resolves equal distances in retained center order, not row order.
        midpoint = (sample[selected[0]] + sample[selected[1]]) / 2
        tied = priors.evaluate_frozen_function(result.function, torch.tensor(midpoint[None, :]))
        np.testing.assert_array_equal(tied.numpy(), [expected[0]])
    frozen_sample = result.calibration.sample
    sample[:] = -100
    assert result.calibration.sample == frozen_sample
    result.validate(context.freeze())


def test_joint_discretization_json_binds_centers_targets_and_all_matrix_families() -> None:
    for family in ("gaussian", "importance", "factorized", "kernel", "transformed"):
        context = priors.JointSamplingContext(np.random.default_rng(83), mode="local")
        plan = priors.sample_joint_function(
            context,
            "disc",
            input_width=2,
            output_width=1,
            calibration_rows=3,
            prior=priors.JointFunctionPrior(
                functions="disc",
                discretization_centers=2,
                matrix=priors.JointMatrixPrior(families=(family,)),
            ),
        )
        result = priors.materialize_joint_function(
            plan,
            context.freeze(),
            rng=np.random.default_rng(84),
            fit_rng=np.random.default_rng(85),
            calibration=np.array([[0.0, 1.0], [2.0, 3.0], [4.0, 5.0]]),
            calibration_id="disc-json",
        )
        payload = json.loads(json.dumps(asdict(result)))
        assert (
            priors.joint_function_realization_from_dict(payload, decisions=context.freeze())
            == result
        )
        for path in (
            ("targets", 0, 0),
            ("function", "centers", 0, 0),
            ("function", "linear", "matrix", 0, 0),
        ):
            changed = copy.deepcopy(payload)
            node = changed
            for key in path[:-1]:
                node = node[key]
            node[path[-1]] += 0.25
            with pytest.raises(ValueError, match="construction"):
                priors.joint_function_realization_from_dict(changed, decisions=context.freeze())
        changed = copy.deepcopy(payload)
        changed["permutation"] = [changed["permutation"][0]] * 3
        with pytest.raises(ValueError, match="permutation"):
            priors.joint_function_realization_from_dict(changed, decisions=context.freeze())
        changed = copy.deepcopy(payload)
        changed["calibration"]["sample"][0][0] += 1
        with pytest.raises(ValueError, match="digest"):
            priors.joint_function_realization_from_dict(changed, decisions=context.freeze())


def test_joint_discretization_failed_target_fit_rolls_back_both_streams_then_recovers() -> None:
    context = priors.JointSamplingContext(np.random.default_rng(80), mode="local")
    plan = priors.sample_joint_function(
        context,
        "disc",
        input_width=2,
        output_width=1,
        calibration_rows=3,
        prior=priors.JointFunctionPrior(
            functions="disc",
            discretization_centers=2,
            matrix=priors.JointMatrixPrior(families=("gaussian",)),
        ),
    )
    decisions = context.freeze()
    raw, fit = np.random.default_rng(82), np.random.default_rng(81)
    before = pickle.dumps((raw.bit_generator.state, fit.bit_generator.state))
    # Finite calibration is admitted; the selected centers overflow only when
    # multiplied by the already drawn and normalized Gaussian matrix.
    huge = np.full((3, 2), 1.7e308)
    with pytest.raises(ValueError, match="finite"):
        priors.materialize_joint_function(
            plan,
            decisions,
            rng=raw,
            fit_rng=fit,
            calibration=huge,
            calibration_id="failed-fit",
        )
    assert pickle.dumps((raw.bit_generator.state, fit.bit_generator.state)) == before
    assert context.freeze() == decisions
    sample = np.array([[0.0, 1.0], [2.0, 3.0], [4.0, 5.0]])
    recovered = priors.materialize_joint_function(
        plan,
        decisions,
        rng=raw,
        fit_rng=fit,
        calibration=sample,
        calibration_id="recovered-fit",
    )
    fresh = priors.materialize_joint_function(
        plan,
        decisions,
        rng=np.random.default_rng(82),
        fit_rng=np.random.default_rng(81),
        calibration=sample,
        calibration_id="recovered-fit",
    )
    assert recovered == fresh


def test_joint_discretization_admission_precedes_oversized_permutation_and_sample_copy() -> None:
    context = priors.JointSamplingContext(np.random.default_rng(80), mode="local")
    plan = priors.sample_joint_function(
        context,
        "disc",
        input_width=2,
        output_width=1,
        calibration_rows=1_000_000,
        prior=priors.JointFunctionPrior(functions="disc", discretization_centers=2),
    )
    raw, fit = np.random.default_rng(82), np.random.default_rng(81)
    before = pickle.dumps((raw.bit_generator.state, fit.bit_generator.state))
    sample = np.broadcast_to(np.array([[np.nan, np.nan]]), (1_000_000, 2))
    with pytest.raises(ValueError, match="limit"):
        priors.materialize_joint_function(
            plan,
            context.freeze(),
            rng=raw,
            fit_rng=fit,
            calibration=sample,
            calibration_id="oversized-discretization-fit",
        )
    assert pickle.dumps((raw.bit_generator.state, fit.bit_generator.state)) == before
