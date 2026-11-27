"""Known prototype raw budgets reject before any generator copy or snapshot."""

from dataclasses import replace
from unittest.mock import patch

import numpy as np
import pytest

from r0.modules import priors as p


@pytest.mark.parametrize("batch", [False, True])
@pytest.mark.parametrize("cap", ["numbers", "draws", "working_bytes"])
def test_prototype_known_raw_caps_precede_rng_copy(batch: bool, cap: str):
    limits = {
        "numbers": p.RawDrawLimits(numbers=7 if not batch else 39),
        "draws": p.RawDrawLimits(draws=1),
        "working_bytes": p.RawDrawLimits(working_bytes=1000),
    }[cap]
    # A ball adds a separate radius record, so even bank-only requires two draws.
    plan = p.PrototypeRootPlan(p.JointRootBase("ball", 2), 8, 0.5, raw_limits=limits)

    def construct():
        if batch:
            return p.sample_prototype_batch_reference(
                plan, prototype_rng=np.random.default_rng(1), weights_rng=np.random.default_rng(2)
            )
        return p.sample_prototype_root(plan, rng=np.random.default_rng(1))

    with (
        patch("copy.deepcopy", side_effect=AssertionError("copied before raw admission")),
        pytest.raises(ValueError, match="raw.*limit"),
    ):
        construct()


@pytest.mark.parametrize("batch", [False, True])
def test_prototype_replay_caps_precede_restore(batch: bool):
    plan = p.PrototypeRootPlan(p.JointRootBase("gaussian", 2), 8, 0.5)
    sample = (
        p.sample_prototype_batch_reference(
            plan, prototype_rng=np.random.default_rng(1), weights_rng=np.random.default_rng(2)
        )
        if batch
        else p.sample_prototype_root(plan, rng=np.random.default_rng(1))
    )
    invalid = replace(plan, raw_limits=p.RawDrawLimits(numbers=7))
    with (
        patch("copy.deepcopy", side_effect=AssertionError("restored before raw admission")),
        pytest.raises(ValueError, match="raw.*limit"),
    ):
        replace(sample, plan=invalid)
