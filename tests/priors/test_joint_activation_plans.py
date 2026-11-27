"""Source activation plans bind their realized mechanisms to shared decisions."""

import json
from dataclasses import asdict

import numpy as np
import pytest
import torch

from r0.modules import priors


def test_joint_activation_plans_share_fixed_primitives_and_standard_scales() -> None:
    context = priors.JointSamplingContext(np.random.default_rng(0), mode="global")
    plans = (
        priors.sample_joint_activation(context, "matrix", selector="plain"),
        priors.sample_joint_activation(context, "node", selector="standard"),
        priors.sample_joint_activation(context, "converter", selector="standard"),
    )
    record = context.freeze()
    assert tuple(plan.operation for plan in plans) == (("matrix",), ("node",), ("converter",))
    assert all(plan.activation == priors.CausalActivation("softplus") for plan in plans)
    assert plans[0].scale is None
    assert plans[1].scale == plans[2].scale
    assert [d.preference for d in record.categorical.draws] == [0, 1, 2, 1, 2, 1]
    assert [d.preference for d in record.numerical.draws] == [0, 0]
    assert [e.role for e in record.events] == [
        "selector",
        "primitive",
        "selector",
        "scale",
        "primitive",
        "selector",
        "scale",
        "primitive",
    ]
    for plan in plans:
        plan.validate(record)
        decoded = priors.joint_activation_plan_from_dict(
            json.loads(json.dumps(asdict(plan))), decisions=record
        )
        assert decoded == plan


def test_joint_standard_activation_fits_the_planned_function_without_resampling() -> None:
    context = priors.JointSamplingContext(np.random.default_rng(0), mode="global")
    plan = priors.sample_joint_activation(context, "node")
    decisions = context.freeze()
    assert plan.activation.kind == "leaky_relu"
    sample = np.array([[-2.0, 0.0], [0.0, 4.0], [2.0, 8.0]])
    fitted = priors.fit_joint_activation(
        plan,
        decisions,
        sample,
        rng=np.random.default_rng(99),
        calibration_id="independent-node-activation",
    )
    # Known seed-99 row choices are [2,1]; the primitive and scale were already frozen.
    assert plan.scale is not None
    standardized = (sample - np.array([0.0, 4.0])) / (np.sqrt(np.array([8.0, 32.0]) / 3) + 1e-4)
    argument = plan.scale * (standardized - standardized[[2, 1], [0, 1]])
    primitive = np.where(argument >= 0, argument, 0.01 * argument)
    expected = (primitive - primitive.mean(axis=0)) / (primitive.std(axis=0) + 1e-4)
    torch.testing.assert_close(
        fitted.apply(torch.tensor(sample, dtype=torch.float64)),
        torch.tensor(expected, dtype=torch.float64),
        rtol=1e-12,
        atol=1e-12,
    )
    assert context.freeze() == decisions


def test_joint_activation_decoder_rejects_coerced_parameter_pairs() -> None:
    context = priors.JointSamplingContext(np.random.default_rng(4), mode="global")
    plan = priors.sample_joint_activation(context, "matrix", selector="plain")
    assert plan.activation.kind == "integer_power"
    raw = asdict(plan)
    key, value = plan.activation.parameters[0]
    # Iterating this malformed mapping would recreate the valid pair from its keys.
    raw["activation"]["parameters"] = [{key: None, value: None}]
    with pytest.raises(ValueError, match="pair.*array"):
        priors.joint_activation_plan_from_dict(raw, decisions=context.freeze())
