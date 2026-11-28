"""Product construction preserves left-to-right draws and shared fit inputs."""

import copy
import itertools
import json
import pickle
from dataclasses import asdict

import numpy as np
import pytest
import torch

from r0.modules import priors


def test_joint_product_fits_both_children_on_the_same_input_in_source_order() -> None:
    context = priors.JointSamplingContext(np.random.default_rng(24), mode="local")
    plan = priors.sample_joint_function(
        context,
        "prod",
        input_width=2,
        output_width=2,
        calibration_rows=3,
        prior=priors.JointFunctionPrior(
            functions="prod",
            frequencies=4,
            matrix=priors.JointMatrixPrior(families=("gaussian",)),
        ),
    )
    assert tuple(child.family for child in plan.children) == ("disc", "disc")
    sample = np.array([[0.0, 1.0], [2.0, -1.0], [-2.0, 4.0]])
    reference_raw, reference_fit = np.random.default_rng(99), np.random.default_rng(98)
    left = priors.materialize_joint_function(
        plan.children[0],
        context.freeze(),
        rng=reference_raw,
        fit_rng=reference_fit,
        calibration=sample,
        calibration_id="shared-product-fit",
    )
    right = priors.materialize_joint_function(
        plan.children[1],
        context.freeze(),
        rng=reference_raw,
        fit_rng=reference_fit,
        calibration=sample,
        calibration_id="shared-product-fit",
    )
    raw, fit = np.random.default_rng(99), np.random.default_rng(98)
    result = priors.materialize_joint_function(
        plan,
        context.freeze(),
        rng=raw,
        fit_rng=fit,
        calibration=sample,
        calibration_id="shared-product-fit",
    )
    assert isinstance(result, priors.JointProductFunctionRealization)
    assert result.left == left
    assert result.right == right
    assert isinstance(right, priors.JointDiscretizationFunctionRealization)
    assert result.calibration == right.calibration
    assert raw.bit_generator.state == reference_raw.bit_generator.state
    assert fit.bit_generator.state == reference_fit.bit_generator.state
    query = torch.tensor([[0.0, 0.0], [1.0, 2.0], [10.0, -3.0]])
    expected = priors.evaluate_frozen_function(
        left.function, query
    ) * priors.evaluate_frozen_function(right.function, query)
    torch.testing.assert_close(priors.evaluate_frozen_function(result.function, query), expected)
    assert not np.array_equal(
        priors.evaluate_frozen_function(left.function, torch.tensor(sample)).numpy(), sample
    )
    assert result.calibration is not None
    np.testing.assert_array_equal(result.calibration.sample, sample)
    result.validate(context.freeze())


def test_joint_product_json_covers_all_25_ordered_cheap_family_pairs() -> None:
    seeds = (
        0,
        1,
        2,
        3,
        4,
        5,
        8,
        10,
        11,
        12,
        16,
        17,
        18,
        22,
        24,
        31,
        37,
        39,
        41,
        46,
        62,
        63,
        67,
        73,
        87,
    )
    seen: set[tuple[str, ...]] = set()
    for seed in seeds:
        context = priors.JointSamplingContext(np.random.default_rng(seed), mode="local")
        plan = priors.sample_joint_function(
            context,
            "prod",
            input_width=2,
            output_width=2,
            calibration_rows=3,
            prior=priors.JointFunctionPrior(
                functions="prod",
                frequencies=4,
                matrix=priors.JointMatrixPrior(families=("gaussian",)),
            ),
        )
        families = tuple(child.family for child in plan.children)
        seen.add(families)
        fitted = any(family in ("tree", "disc") for family in families)
        result = priors.materialize_joint_function(
            plan,
            context.freeze(),
            rng=np.random.default_rng(99),
            fit_rng=np.random.default_rng(98) if fitted else None,
            calibration=np.array([[0.0, 1.0], [2.0, -1.0], [-2.0, 4.0]]) if fitted else None,
            calibration_id="shared-product-fit" if fitted else "",
        )
        payload = json.loads(json.dumps(asdict(result)))
        assert (
            priors.joint_function_realization_from_dict(payload, decisions=context.freeze())
            == result
        )
        changed = copy.deepcopy(payload)
        changed["function"]["left"] = copy.deepcopy(changed["function"]["right"])
        with pytest.raises(ValueError, match="constructions"):
            priors.joint_function_realization_from_dict(changed, decisions=context.freeze())
        changed = copy.deepcopy(payload)
        changed["left"], changed["right"] = changed["right"], changed["left"]
        with pytest.raises(ValueError, match="ordered owned plans"):
            priors.joint_function_realization_from_dict(changed, decisions=context.freeze())
        if fitted:
            changed = copy.deepcopy(payload)
            changed["calibration"]["identity"] = "different-fit"
            with pytest.raises(ValueError, match="same retained calibration"):
                priors.joint_function_realization_from_dict(changed, decisions=context.freeze())
    assert seen == set(itertools.product(("gp", "tree", "disc", "lin", "quad"), repeat=2))


def test_joint_product_rolls_back_successful_left_child_when_right_fit_fails() -> None:
    context = priors.JointSamplingContext(np.random.default_rng(10), mode="local")
    plan = priors.sample_joint_function(
        context,
        "prod",
        input_width=2,
        output_width=2,
        calibration_rows=4,
        prior=priors.JointFunctionPrior(
            functions="prod",
            frequencies=4,
            discretization_centers=4,
            matrix=priors.JointMatrixPrior(families=("gaussian",)),
        ),
    )
    assert tuple(child.family for child in plan.children) == ("tree", "disc")
    sample = np.array([[1.0, 1.0], [1.0, -1.0], [-1.0, 1.0], [-1.0, -1.0]]) * 1.7e308
    decisions = context.freeze()
    reference_raw, reference_fit = np.random.default_rng(99), np.random.default_rng(98)
    # Establish that this fixture reaches the right child after a valid left fit.
    left = priors.materialize_joint_function(
        plan.children[0],
        decisions,
        rng=reference_raw,
        fit_rng=reference_fit,
        calibration=sample,
        calibration_id="failing-right-fit",
    )
    assert isinstance(left, priors.JointTreeFunctionRealization)
    with pytest.raises(ValueError, match="finite"):
        priors.materialize_joint_function(
            plan.children[1],
            decisions,
            rng=reference_raw,
            fit_rng=reference_fit,
            calibration=sample,
            calibration_id="failing-right-fit",
        )
    raw, fit = np.random.default_rng(99), np.random.default_rng(98)
    before = pickle.dumps((raw.bit_generator.state, fit.bit_generator.state))
    with pytest.raises(ValueError, match="finite"):
        priors.materialize_joint_function(
            plan,
            decisions,
            rng=raw,
            fit_rng=fit,
            calibration=sample,
            calibration_id="failing-right-fit",
        )
    assert pickle.dumps((raw.bit_generator.state, fit.bit_generator.state)) == before
    assert context.freeze() == decisions
    recovered = priors.materialize_joint_function(
        plan,
        decisions,
        rng=raw,
        fit_rng=fit,
        calibration=sample / 1.7e308,
        calibration_id="recovered-product",
    )
    fresh = priors.materialize_joint_function(
        plan,
        decisions,
        rng=np.random.default_rng(99),
        fit_rng=np.random.default_rng(98),
        calibration=sample / 1.7e308,
        calibration_id="recovered-product",
    )
    assert recovered == fresh


def test_joint_product_admission_aggregates_children_that_individually_fit() -> None:
    context = priors.JointSamplingContext(np.random.default_rng(37), mode="local")
    plan = priors.sample_joint_function(
        context,
        "prod",
        input_width=2,
        output_width=2,
        calibration_rows=3,
        prior=priors.JointFunctionPrior(
            functions="prod",
            frequencies=4,
            matrix=priors.JointMatrixPrior(families=("gaussian",)),
        ),
    )
    assert tuple(child.family for child in plan.children) == ("lin", "lin")
    limits = priors.CausalLimits(parameters=128)
    for child in plan.children:
        priors.materialize_joint_function(
            child, context.freeze(), rng=np.random.default_rng(99), limits=limits
        )
    rng = np.random.default_rng(99)
    before = pickle.dumps(rng.bit_generator.state)
    with pytest.raises(ValueError, match="aggregate product parameter limit"):
        priors.materialize_joint_function(plan, context.freeze(), rng=rng, limits=limits)
    assert pickle.dumps(rng.bit_generator.state) == before
