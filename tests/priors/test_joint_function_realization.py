"""Frozen equations stay bound to sampled source constructions."""

import copy
import json
from dataclasses import asdict

import numpy as np
import pytest
import torch

from r0.modules import priors


def test_joint_linear_realization_retains_the_source_output_by_input_matrix() -> None:
    context = priors.JointSamplingContext(np.random.default_rng(50))
    plan = priors.sample_joint_function(
        context,
        "linear",
        input_width=3,
        output_width=2,
        calibration_rows=8,
        prior=priors.JointFunctionPrior(
            functions="lin", matrix=priors.JointMatrixPrior(families=("gaussian",))
        ),
    )
    decisions = context.freeze()
    realization = priors.materialize_joint_function(plan, decisions, rng=np.random.default_rng(51))
    assert isinstance(realization, priors.JointMatrixFunctionRealization)
    assert realization.plan == plan
    assert realization.matrix.plan == plan.matrices[0]
    assert realization.permutation == ()
    # Basis rows expose the retained source matrix orientation without invoking
    # another function evaluator to manufacture the expected values.
    matrix = realization.matrix.materialize()[0]
    expected = torch.tensor(
        [
            [matrix[0][0], matrix[1][0]],
            [matrix[0][1], matrix[1][1]],
            [matrix[0][2], matrix[1][2]],
        ],
        dtype=torch.float64,
    )
    actual = priors.evaluate_frozen_function(realization.function, torch.eye(3))
    torch.testing.assert_close(actual, expected)
    realization.validate(decisions)


def test_joint_quadratic_realization_retains_full_permutation_before_the_matrix_batch() -> None:
    for width in (3, 21):
        context = priors.JointSamplingContext(np.random.default_rng(52))
        plan = priors.sample_joint_function(
            context,
            "quadratic",
            input_width=width,
            output_width=2,
            calibration_rows=8,
            prior=priors.JointFunctionPrior(
                functions="quad", matrix=priors.JointMatrixPrior(families=("gaussian",))
            ),
        )
        source_rng = np.random.default_rng(53)
        permutation = tuple(map(int, source_rng.permutation(width))) if width > 20 else ()
        indices: tuple[int, ...] = permutation[:20] if width > 20 else tuple(range(width))
        k = len(indices) + 1
        gaussian = source_rng.normal(size=(2, k, k))
        realization = priors.materialize_joint_function(
            plan, context.freeze(), rng=np.random.default_rng(53)
        )
        assert isinstance(realization, priors.JointMatrixFunctionRealization)
        assert realization.permutation == permutation
        assert isinstance(realization.function, priors.FrozenQuadraticFunction)
        assert realization.function.indices == indices
        for i, matrix in enumerate(realization.matrix.matrices):
            assert isinstance(matrix.source, priors.GaussianWeights)
            assert matrix.source.draws == tuple(tuple(map(float, row)) for row in gaussian[i])
        matrices = realization.matrix.materialize()
        values = torch.zeros((2, width), dtype=torch.float64)
        values[1, next(iter(indices))] = 1
        expected = torch.tensor(
            [
                [q[-1][-1] for q in matrices],
                [q[0][0] + q[0][-1] + q[-1][0] + q[-1][-1] for q in matrices],
            ],
            dtype=torch.float64,
        )
        torch.testing.assert_close(
            priors.evaluate_frozen_function(realization.function, values), expected
        )
        realization.validate(context.freeze())


def test_joint_matrix_function_json_rejects_equations_changed_from_all_matrix_constructions() -> (
    None
):
    for family in ("lin", "quad"):
        for matrix in ("gaussian", "importance", "factorized", "kernel", "transformed"):
            context = priors.JointSamplingContext(np.random.default_rng(54))
            plan = priors.sample_joint_function(
                context,
                "function",
                input_width=3,
                output_width=2,
                calibration_rows=8,
                prior=priors.JointFunctionPrior(
                    functions=family, matrix=priors.JointMatrixPrior(families=(matrix,))
                ),
            )
            realization = priors.materialize_joint_function(
                plan, context.freeze(), rng=np.random.default_rng(55)
            )
            payload = json.loads(json.dumps(asdict(realization)))
            assert (
                priors.joint_function_realization_from_dict(payload, decisions=context.freeze())
                == realization
            )
            if family == "lin":
                payload["function"]["matrix"][0][0] += 0.25
            else:
                payload["function"]["matrices"][0][0][0] += 0.25
            with pytest.raises(ValueError, match="construction"):
                priors.joint_function_realization_from_dict(payload, decisions=context.freeze())


def test_joint_function_realization_admission_precedes_large_permutation_and_rng_changes() -> None:
    for family, width, limits in (
        ("quad", 10**9, priors.CausalLimits()),
        ("lin", 3, priors.CausalLimits(parameters=1)),
        ("quad", 3, priors.CausalLimits(working_bytes=64)),
    ):
        context = priors.JointSamplingContext(np.random.default_rng(56))
        plan = priors.sample_joint_function(
            context,
            "function",
            input_width=width,
            output_width=2,
            calibration_rows=8,
            prior=priors.JointFunctionPrior(
                functions=family, matrix=priors.JointMatrixPrior(families=("gaussian",))
            ),
        )
        record = context.freeze()
        rng = np.random.default_rng(57)
        state = copy.deepcopy(rng.bit_generator.state)
        with pytest.raises(ValueError, match="limit"):
            priors.materialize_joint_function(plan, record, rng=rng, limits=limits)
        assert rng.bit_generator.state == state
        assert context.freeze() == record
