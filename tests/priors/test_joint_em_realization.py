"""EM-style fits retain anchor selection, Gaussian construction and assignments."""

import copy
import json
import math
import pickle
from dataclasses import asdict

import numpy as np
import pytest
import torch

from r0.modules import priors


def test_joint_em_fits_source_assignments_before_its_output_matrix() -> None:
    context = priors.JointSamplingContext(np.random.default_rng(90), mode="local")
    plan = priors.sample_joint_function(
        context,
        "em",
        input_width=2,
        output_width=2,
        calibration_rows=2,
        prior=priors.JointFunctionPrior(
            functions="em",
            matrix=priors.JointMatrixPrior(families=("gaussian",)),
        ),
    )
    parameters = dict(plan.parameters)
    count = int(parameters["components"])
    sample = np.array([[0.0, 1.0], [2.0, -1.0]])
    raw_oracle, fit_oracle = np.random.default_rng(91), np.random.default_rng(92)
    rows = fit_oracle.integers(2, size=count)
    jitter = raw_oracle.normal(size=(count, 2))
    spread_gaussian = raw_oracle.normal(size=count)
    gaussian = raw_oracle.normal(size=(1, 2, count))
    raw_oracle.normal(size=(1, 2, count))
    raw, fit = np.random.default_rng(91), np.random.default_rng(92)
    result = priors.materialize_joint_function(
        plan,
        context.freeze(),
        rng=raw,
        fit_rng=fit,
        calibration=sample,
        calibration_id="independent-em-fit",
    )
    assert isinstance(result, priors.JointEMFunctionRealization)
    np.testing.assert_array_equal(result.anchor_rows, rows)
    np.testing.assert_array_equal(result.center_gaussian, jitter)
    np.testing.assert_array_equal(result.spread_gaussian, spread_gaussian)
    centers = sample[rows] + jitter
    scales = np.exp(float(parameters["spread_factor"]) * spread_gaussian)
    np.testing.assert_array_equal(result.function.centers, centers)
    np.testing.assert_array_equal(result.function.scales, scales)
    source = result.matrix.matrices[0].source
    assert isinstance(source, priors.GaussianWeights)
    np.testing.assert_array_equal(source.draws, gaussian[0])
    assert pickle.dumps(raw.bit_generator.state) == pickle.dumps(raw_oracle.bit_generator.state)
    assert pickle.dumps(fit.bit_generator.state) == pickle.dumps(fit_oracle.bit_generator.state)
    p, q = float(parameters["distance_p"]), float(parameters["distance_q"])
    expected: list[list[float]] = []
    wrong_constants: list[list[float]] = []
    for row in sample:
        distances = [
            sum(abs(float(a - b)) ** p for a, b in zip(row, center, strict=True)) ** (1 / p)
            for center in centers
        ]
        penalties = [
            (distance / scale) ** q for distance, scale in zip(distances, scales, strict=True)
        ]
        constants = [-0.5 * math.log(2 * math.pi * float(scale) ** 2) for scale in scales]
        logits = [
            constant - penalty for constant, penalty in zip(constants, penalties, strict=True)
        ]
        masses = [math.exp(value - max(logits)) for value in logits]
        expected.append([value / sum(masses) for value in masses])
        wrong_logits = [
            2 * constant - penalty for constant, penalty in zip(constants, penalties, strict=True)
        ]
        wrong_masses = [math.exp(value - max(wrong_logits)) for value in wrong_logits]
        wrong_constants.append([value / sum(wrong_masses) for value in wrong_masses])
    np.testing.assert_allclose(result.assignments, expected, rtol=1e-12, atol=1e-15)
    assert not np.allclose(expected, wrong_constants)
    matrix = result.matrix.materialize()[0]
    expected_outputs = [
        [
            sum(prob * coefficient for prob, coefficient in zip(row, output, strict=True))
            for output in matrix
        ]
        for row in expected
    ]
    actual = priors.evaluate_frozen_function(result.function, torch.tensor(sample))
    np.testing.assert_allclose(actual.numpy(), expected_outputs, rtol=1e-12, atol=1e-15)
    result.validate(context.freeze())


def test_joint_em_json_binds_raw_anchors_spreads_assignments_and_all_matrix_families() -> None:
    for family in ("gaussian", "importance", "factorized", "kernel", "transformed"):
        context = priors.JointSamplingContext(np.random.default_rng(90), mode="local")
        plan = priors.sample_joint_function(
            context,
            "em",
            input_width=2,
            output_width=2,
            calibration_rows=3,
            prior=priors.JointFunctionPrior(
                functions="em",
                matrix=priors.JointMatrixPrior(families=(family,)),
            ),
        )
        result = priors.materialize_joint_function(
            plan,
            context.freeze(),
            rng=np.random.default_rng(91),
            fit_rng=np.random.default_rng(92),
            calibration=np.array([[0.0, 1.0], [2.0, -1.0], [-2.0, 4.0]]),
            calibration_id="em-json",
        )
        payload = json.loads(json.dumps(asdict(result)))
        assert (
            priors.joint_function_realization_from_dict(payload, decisions=context.freeze())
            == result
        )
        for path in (
            ("assignments", 0, 0),
            ("center_gaussian", 0, 0),
            ("spread_gaussian", 0),
            ("function", "scales", 0),
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
        changed["anchor_rows"][0] = True
        with pytest.raises(ValueError, match="row indices"):
            priors.joint_function_realization_from_dict(changed, decisions=context.freeze())
        changed = copy.deepcopy(payload)
        changed["calibration"]["sample"][0][0] += 1
        with pytest.raises(ValueError, match="digest"):
            priors.joint_function_realization_from_dict(changed, decisions=context.freeze())


def test_joint_em_nonfinite_assignment_fit_rolls_back_both_streams_and_recovers() -> None:
    context = priors.JointSamplingContext(np.random.default_rng(90), mode="local")
    plan = priors.sample_joint_function(
        context,
        "em",
        input_width=1,
        output_width=1,
        calibration_rows=17,
        prior=priors.JointFunctionPrior(
            functions="em",
            matrix=priors.JointMatrixPrior(families=("gaussian",)),
        ),
    )
    decisions = context.freeze()
    raw, fit = np.random.default_rng(91), np.random.default_rng(92)
    before = pickle.dumps((raw.bit_generator.state, fit.bit_generator.state))
    # At most 16 selected anchors cannot cover 17 distinct, widely separated rows.
    # A missing row has infinite penalties to every anchor, so softmax is undefined.
    sample = np.arange(-8, 9, dtype=np.float64)[:, None] * 1e307
    assert np.isfinite(sample).all()
    with pytest.raises(ValueError, match="nonfinite EM assignment"):
        priors.materialize_joint_function(
            plan,
            decisions,
            rng=raw,
            fit_rng=fit,
            calibration=sample,
            calibration_id="failed-em-fit",
        )
    assert pickle.dumps((raw.bit_generator.state, fit.bit_generator.state)) == before
    assert context.freeze() == decisions
    recovered = priors.materialize_joint_function(
        plan,
        decisions,
        rng=raw,
        fit_rng=fit,
        calibration=sample / 1e307,
        calibration_id="recovered-em-fit",
    )
    fresh = priors.materialize_joint_function(
        plan,
        decisions,
        rng=np.random.default_rng(91),
        fit_rng=np.random.default_rng(92),
        calibration=sample / 1e307,
        calibration_id="recovered-em-fit",
    )
    assert recovered == fresh


def test_joint_em_admission_precedes_large_assignment_arrays_and_sample_validation() -> None:
    context = priors.JointSamplingContext(np.random.default_rng(90), mode="local")
    plan = priors.sample_joint_function(
        context,
        "em",
        input_width=2,
        output_width=1,
        calibration_rows=1_000_000,
        prior=priors.JointFunctionPrior(functions="em"),
    )
    raw, fit = np.random.default_rng(91), np.random.default_rng(92)
    before = pickle.dumps((raw.bit_generator.state, fit.bit_generator.state))
    sample = np.broadcast_to(np.array([[np.nan, np.nan]]), (1_000_000, 2))
    with pytest.raises(ValueError, match="limit"):
        priors.materialize_joint_function(
            plan,
            context.freeze(),
            rng=raw,
            fit_rng=fit,
            calibration=sample,
            calibration_id="oversized-em-fit",
        )
    assert pickle.dumps((raw.bit_generator.state, fit.bit_generator.state)) == before
