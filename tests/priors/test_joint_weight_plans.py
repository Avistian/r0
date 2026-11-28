"""Shared polynomial hyperparameters remain separate from vector realizations."""

import copy
import json
from dataclasses import asdict

import numpy as np
import pytest

from r0.modules import priors


def test_joint_weight_plans_share_outer_preferences_across_widths_and_materialize() -> None:
    context = priors.JointSamplingContext(np.random.default_rng(28), mode="global")
    small = priors.sample_joint_weights(context, "matrix-row", width=2)
    large = priors.sample_joint_weights(context, "node-channels", width=7)
    decisions = context.freeze()
    assert [d.preference for d in decisions.numerical.draws] == [0, 1, 0, 1]
    assert decisions.numerical.draws[0].units == decisions.numerical.draws[2].units
    assert small.decays != large.decays
    assert small.scales == large.scales
    for plan in (small, large):
        assert (
            priors.joint_weight_plan_from_dict(
                json.loads(json.dumps(asdict(plan))), decisions=decisions
            )
            == plan
        )
        rng = np.random.default_rng(17)
        control = copy.deepcopy(rng)
        vector = priors.materialize_joint_weights(plan, decisions, rng=rng)[0]
        assert isinstance(vector, priors.PolynomialWeightVector)
        assert vector.draws == tuple(control.normal(size=plan.width))
        assert vector.permutation == tuple(control.permutation(plan.width))
        assert sum(v * v for v in vector.values()) == pytest.approx(plan.width)
        np.testing.assert_array_equal(rng.random(8), control.random(8))


def test_joint_weight_batches_keep_source_parameter_order() -> None:
    context = priors.JointSamplingContext(np.random.default_rng(28), mode="local")
    plan = priors.sample_joint_weights(context, "matrix-rows", width=4, batch=3)
    record = context.freeze()
    assert [e.role for e in record.events] == [
        "decay:0",
        "decay:1",
        "decay:2",
        "scale:0",
        "scale:1",
        "scale:2",
    ]
    assert [d.preference for d in record.numerical.draws] == [0, 0, 0, 1, 1, 1]
    assert plan.batch == 3
    assert len(plan.decays) == len(plan.scales) == 3
    vectors = priors.materialize_joint_weights(plan, record, rng=np.random.default_rng(17))
    assert len(vectors) == 3
    assert all(len(vector.draws) == 4 for vector in vectors)


def test_source_weight_batches_retain_arrays_under_strict_global_contracts() -> None:
    context = priors.JointSamplingContext(
        np.random.default_rng(28), mode="global", global_policy="strict-contract-v1"
    )
    first = priors.sample_joint_weights(
        context, "first", width=4, batch=3, scalarization="source-weight-batches-v1"
    )
    second = priors.sample_joint_weights(
        context, "second", width=4, batch=3, scalarization="source-weight-batches-v1"
    )
    record = context.freeze()
    assert first.decays == second.decays
    assert len(set(first.decays)) > 1
    assert [e.role for e in record.events] == ["decay", "scale", "decay", "scale"]
    assert [d.request.shape for d in record.numerical.draws] == [(3,)] * 4
    with pytest.raises(ValueError, match="contract"):
        priors.sample_joint_weights(
            context, "incompatible", width=4, batch=2, scalarization="source-weight-batches-v1"
        )
    assert context.freeze() == record
    assert priors.joint_weight_plan_from_dict(asdict(first), decisions=record) == first
