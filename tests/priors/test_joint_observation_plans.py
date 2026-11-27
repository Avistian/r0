"""Feature groups choose eligible graph nodes before channel widths are known."""

import copy
from dataclasses import replace

import numpy as np
import pytest

from r0.modules import priors


@pytest.mark.parametrize("subsample", [False, True])
@pytest.mark.parametrize("engine", ["PCG64", "PCG64DXSM", "Philox", "SFC64", "MT19937"])
def test_group_assignments_retain_raw_order_and_match_numpy_oracle(
    subsample: bool, engine: str
) -> None:
    context = priors.JointSamplingContext(np.random.default_rng(44), mode="global")
    groups = (priors.FeatureGroup("x", 5), priors.FeatureGroup("y", 2))
    plan = priors.sample_joint_observations(
        context, "observations", nodes=4, groups=groups, subsample=subsample
    )
    decisions = context.freeze()
    plan.validate(decisions)
    if subsample:
        assert len(set(plan.eligible_counts)) == 1
        assert decisions.numerical.preferences[0].request.name == "n_feature_nodes"
    else:
        assert plan.eligible_counts == (4, 4)
        assert decisions.numerical.draws == ()
    rng = np.random.Generator(getattr(np.random, engine)(92))
    control = copy.deepcopy(rng)
    initial = copy.deepcopy(rng)
    result = priors.materialize_joint_observations(plan, decisions, rng=rng)
    expected: list[tuple[int, ...]] = []
    for group, count in zip(groups, plan.eligible_counts, strict=True):
        nodes = control.permutation(4)[:count] if subsample else np.arange(4)
        expected.append(tuple(int(nodes[i]) for i in control.integers(count, size=group.count)))
    assert result.assignments == tuple(expected)
    result.validate(decisions)
    assert result.replay(decisions, copy.deepcopy(initial)) == result
    tampered = replace(result, raw=(replace(result.raw[0], state_before="0" * 64), *result.raw[1:]))
    unchanged = copy.deepcopy(initial)
    with pytest.raises(ValueError, match="state"):
        tampered.replay(decisions, initial)
    np.testing.assert_array_equal(initial.random(8), unchanged.random(8))
    np.testing.assert_array_equal(rng.random(8), control.random(8))
    corrupted = replace(result, assignments=((99,) * 5, expected[1]))
    with pytest.raises(ValueError, match="assignments"):
        corrupted.validate(decisions)


def test_assignment_budget_failure_does_not_advance_raw_stream() -> None:
    context = priors.JointSamplingContext(np.random.default_rng(44), mode="local")
    plan = priors.sample_joint_observations(
        context, "observations", nodes=4, groups=(priors.FeatureGroup("x", 5),), subsample=True
    )
    rng = np.random.default_rng(92)
    control = copy.deepcopy(rng)
    with pytest.raises(ValueError, match="raw draw.*limit"):
        priors.materialize_joint_observations(
            plan, context.freeze(), rng=rng, raw_limits=priors.RawDrawLimits(draws=1)
        )
    np.testing.assert_array_equal(rng.random(8), control.random(8))
