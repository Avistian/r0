"""Every source root base owns its required width-preserving function plan."""

import copy
import json
from dataclasses import asdict

import numpy as np
import pytest

from r0.modules import priors


def test_joint_root_simple_bases_always_include_a_width_preserving_random_function() -> None:
    for family in ("uniform", "ball", "gaussian"):
        context = priors.JointSamplingContext(np.random.default_rng(40))
        plan = priors.sample_joint_root(
            context,
            "root",
            width=3,
            calibration_rows=8,
            prior=priors.JointRootPrior(
                families=(family,),
                function=priors.JointFunctionPrior(
                    functions="lin", matrix=priors.JointMatrixPrior(families=("gaussian",))
                ),
            ),
        )
        assert plan.family == family
        assert plan.weights is None
        assert plan.covariance_base is None
        assert plan.function.operation == ("root", "function")
        assert (plan.function.input_width, plan.function.output_width) == (3, 3)
        assert plan.function.calibration_rows == 8
        record = context.freeze()
        assert [draw.request.name for draw in record.categorical.draws] == [
            "random_base_points",
            "random_function_fct_type",
            "random_matrix_type",
        ]
        plan.validate(record)


def test_joint_covariance_root_owns_raw_factor_weights_before_base_and_function() -> None:
    seen: set[str | None] = set()
    for seed in range(8):
        for vector_family in ("uniform", "polynomial"):
            context = priors.JointSamplingContext(np.random.default_rng(seed), mode="local")
            plan = priors.sample_joint_root(
                context,
                "root",
                width=3,
                calibration_rows=8,
                prior=priors.JointRootPrior(
                    families=("covariance",),
                    function=priors.JointFunctionPrior(
                        functions="lin",
                        matrix=priors.JointMatrixPrior(
                            families=("gaussian",),
                            vector_family=vector_family,
                        ),
                    ),
                ),
            )
            assert plan.weights is not None
            assert plan.weights.operation == ("root", "weights")
            assert (plan.weights.width, plan.weights.batch) == (3, 1)
            assert plan.weights.family == vector_family
            seen.add(plan.covariance_base)
            record = context.freeze()
            names = [draw.request.name for draw in record.categorical.draws]
            assert names == [
                "random_base_points",
                "random_covariance_base_points",
                "random_function_fct_type",
                "random_matrix_type",
            ]
            expected = ["base", "covariance_base", "family", "family"]
            if vector_family == "polynomial":
                expected[1:1] = ["decay:0", "scale:0"]
            assert [event.role for event in record.events] == expected
            plan.validate(record)
    assert seen == {"uniform", "gaussian"}


def test_joint_root_json_keeps_all_bases_and_rejects_unknown_nested_prior_fields() -> None:
    for family in ("uniform", "ball", "gaussian", "covariance"):
        context = priors.JointSamplingContext(np.random.default_rng(41))
        plan = priors.sample_joint_root(
            context,
            "root",
            width=3,
            calibration_rows=8,
            prior=priors.JointRootPrior(families=(family,)),
        )
        payload = json.loads(json.dumps(asdict(plan)))
        assert priors.joint_root_plan_from_dict(payload, decisions=context.freeze()) == plan
        with pytest.raises(ValueError, match="limit"):
            priors.joint_root_plan_from_dict(
                payload, decisions=context.freeze(), limits=priors.CausalLimits(parameters=1)
            )
        payload["prior"]["function"]["unknown"] = True
        with pytest.raises(ValueError, match="unknown"):
            priors.joint_root_plan_from_dict(payload, decisions=context.freeze())


def test_joint_root_nested_failure_rolls_back_base_weights_and_function_choices() -> None:
    rng = np.random.default_rng(42)
    context = priors.JointSamplingContext(rng)
    priors.sample_joint_activation(context, "existing")
    before, state = context.freeze(), copy.deepcopy(rng.bit_generator.state)
    with pytest.raises(ValueError, match="bounds"):
        priors.sample_joint_root(
            context,
            "root",
            width=2,
            calibration_rows=8,
            prior=priors.JointRootPrior(
                families=("covariance",),
                function=priors.JointFunctionPrior(
                    functions="disc", max_discretization_cardinality=1
                ),
            ),
        )
    assert context.freeze() == before
    assert rng.bit_generator.state == state
    plan = priors.sample_joint_root(
        context,
        "root",
        width=2,
        calibration_rows=8,
        prior=priors.JointRootPrior(function=priors.JointFunctionPrior(functions="lin")),
    )
    plan.validate(context.freeze())


def test_joint_root_json_admits_both_stored_priors_and_rejects_changed_root_width() -> None:
    context = priors.JointSamplingContext(np.random.default_rng(43))
    plan = priors.sample_joint_root(
        context,
        "root",
        width=2,
        calibration_rows=8,
        prior=priors.JointRootPrior(
            families=("uniform",),
            function=priors.JointFunctionPrior(
                functions="lin", matrix=priors.JointMatrixPrior(families=("gaussian",))
            ),
        ),
    )
    payload = json.loads(json.dumps(asdict(plan)))
    decisions = context.freeze()
    # Ten function numeric leaves plus width/rows and two duplicated prior counts.
    assert (
        priors.joint_root_plan_from_dict(
            payload, decisions=decisions, limits=priors.CausalLimits(parameters=14)
        )
        == plan
    )
    with pytest.raises(ValueError, match="limit"):
        priors.joint_root_plan_from_dict(
            payload, decisions=decisions, limits=priors.CausalLimits(parameters=13)
        )
    payload["width"] = 3
    with pytest.raises(ValueError, match="width-preserving"):
        priors.joint_root_plan_from_dict(payload, decisions=decisions)
