"""Latent widths reserve converter channels and bind their named source draws."""

import copy
from dataclasses import replace

import numpy as np
import pytest

from r0.modules import priors


def test_latent_widths_share_named_preferences_and_reserve_converter_channels() -> None:
    rng = np.random.default_rng(82)
    control = copy.deepcopy(rng)
    context = priors.JointSamplingContext(rng, mode="global")
    plan = priors.sample_joint_widths(context, "widths", converter_channels=(2, 0, 3))
    record = context.freeze()
    assert len(set(plan.latent)) == 1
    assert all(1 <= width < 32 for width in plan.latent)
    assert plan.widths == tuple(n + extra for n, extra in zip(plan.latent, (2, 0, 3), strict=True))
    assert len(record.numerical.preferences) == 1
    assert record.numerical.preferences[0].request.name == "node_n_latent_features"
    plan.validate(record)
    oracle = priors.ParameterSampler(control, default_mode="global")
    request = priors.NumericalParameter(
        "node_n_latent_features", 1, 32, scale="log", integer="floor"
    )
    expected = tuple(int(oracle.numerical(request).values[0]) for _ in range(3))
    assert plan.latent == expected
    np.testing.assert_array_equal(rng.random(8), control.random(8))
    changed = replace(plan, latent=tuple(1 if n != 1 else 2 for n in plan.latent))
    with pytest.raises(ValueError, match="named realization"):
        changed.validate(record)


@pytest.mark.parametrize("late", [False, True])
def test_latent_width_limit_failure_rolls_back_named_state_and_registry(late: bool) -> None:
    rng = np.random.default_rng(82)
    control = copy.deepcopy(rng)
    context = priors.JointSamplingContext(rng, mode="local")
    before = context.freeze()
    registry = context.freeze_registry()
    with pytest.raises(ValueError, match="channel limit"):
        priors.sample_joint_widths(
            context,
            "widths",
            converter_channels=(0, 0) if late else (1, 1),
            prior=priors.JointWidthPrior(low=2, high=3),
            limits=priors.CausalLimits(total_channels=3 if late else 2),
        )
    assert context.freeze() == before
    assert context.freeze_registry() == registry
    np.testing.assert_array_equal(rng.random(8), control.random(8))


def test_local_latent_widths_match_scalar_log_uniform_floor_oracle() -> None:
    rng = np.random.default_rng(121)
    control = copy.deepcopy(rng)
    context = priors.JointSamplingContext(rng, mode="local")
    plan = priors.sample_joint_widths(context, "widths", converter_channels=(0,) * 12)
    uniforms = control.random(12)
    expected = tuple(max(1, min(31, int(np.floor(np.exp(u * np.log(32)))))) for u in uniforms)
    assert plan.latent == expected
    assert plan.widths == expected
    np.testing.assert_array_equal(rng.random(8), control.random(8))
