"""Caller pair budgets reject before calibration and either side's generation."""

from dataclasses import replace
from unittest.mock import patch

import pytest
import torch

from r0.modules import priors as p


def network() -> p.NetworkRecipe:
    return p.NetworkRecipe(
        p.LinearRecipe(rows=16, context_rows=8, features=2, signal_features=2, active_features=2),
        p.FrozenNetworkFunction(
            2, (p.NetworkBlock((0,), (p.NetworkAffine(((2.0, -3.0),), (0.5,)),)),)
        ),
    )


@pytest.mark.parametrize(
    ("intervention", "value", "estimated"),
    [
        ("sparsity", 1, True),
        ("sparsity", 1, False),
        ("irrelevant", 3, False),
        ("noise", 0.2, False),
    ],
)
def test_pair_caller_cap_precedes_calibration_and_generation(
    intervention: str, value: int | float, estimated: bool
) -> None:
    def forbidden(*args: object, **kwargs: object) -> None:
        raise AssertionError("caller rejection must precede numerical work")

    with (
        patch.dict(p.fit_network_variance_match.__globals__, draw_root_base=forbidden),
        patch.dict(p.generate_pair.__globals__, generate_task=forbidden),
        pytest.raises(ValueError, match="limit|budget"),
    ):
        p.generate_pair(
            network(),
            intervention=intervention,
            value=value,
            seed=17,
            task_id=3,
            allow_estimated_variance=estimated,
            limits=p.CausalLimits(working_bytes=1),
        )


def test_functional_pair_cap_precedes_generation() -> None:
    def forbidden(*args: object, **kwargs: object) -> None:
        raise AssertionError("functional caller cap must precede generation")

    with (
        patch.dict(p.generate_pair.__globals__, generate_task=forbidden),
        pytest.raises(ValueError, match="limit|budget"),
    ):
        p.generate_pair(
            network().runtime,
            intervention="noise",
            value=0.2,
            seed=17,
            task_id=3,
            limits=p.CausalLimits(working_bytes=1),
        )


@pytest.mark.parametrize("estimated", [False, True])
def test_admission_does_not_rewrite_original_archives(estimated: bool) -> None:
    source = network()
    expected = p.generate_pair(
        source,
        intervention="sparsity",
        value=1,
        seed=17,
        task_id=3,
        allow_estimated_variance=estimated,
    )
    actual = p.generate_pair(
        source,
        intervention="sparsity",
        value=1,
        seed=17,
        task_id=3,
        allow_estimated_variance=estimated,
        limits=replace(source.limits, working_bytes=2 * source.limits.working_bytes),
    )
    assert actual.before.recipe_json == expected.before.recipe_json
    assert actual.after.recipe_json == expected.after.recipe_json
    assert actual.card_json == expected.card_json
    assert torch.equal(actual.after.targets, expected.after.targets)
