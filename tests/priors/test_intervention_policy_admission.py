"""Nested graph and pair budgets bind policy construction before stream transactions."""

from dataclasses import asdict, replace
from unittest.mock import patch

import numpy as np
import pytest

from r0.modules import priors as p


def graph() -> p.FunctionalCausalRecipe:
    return p.FunctionalCausalRecipe(
        p.LinearRecipe(rows=32, context_rows=12, features=1, active_features=1, signal_features=1),
        (
            p.FunctionalCausalNode(
                (), p.FrozenLinearFunction(((1.0,),)), p.JointRootBase("gaussian", 1)
            ),
            p.FunctionalCausalNode((0,), p.FrozenLinearFunction(((-2.0,),))),
            p.FunctionalCausalNode((0, 1), p.FrozenLinearFunction(((3.0, 0.5),))),
            p.FunctionalCausalNode((1, 2), p.FrozenLinearFunction(((4.0, -3.0),))),
        ),
        ((0, 0),),
        ((3, 0),),
    )


def sample(request: p.InterventionPolicyRequest) -> p.SampledInterventionPolicy:
    return p.sample_intervention_policy(
        request,
        selection_rng=np.random.default_rng(1),
        value_rng=np.random.default_rng(2),
        calibration_rng=np.random.default_rng(3),
    )


def test_nested_reference_limit_precedes_any_stream_transaction() -> None:
    base = replace(graph(), limits=p.CausalLimits(working_bytes=512 * 1024))
    base.admit()
    p.CausalInterventionRecipe(base, (p.HardIntervention(1, (0,), (0.0,)),)).admit()
    with pytest.raises(ValueError, match="working"):
        base.admit(evaluation_rows=1024)
    request = p.InterventionPolicyRequest(
        base,
        (3,),
        profile="explicit-policy-r0-v1",
        eligibility=(1,),
        count=1,
        mode="matched",
        reference_rows=1024,
    )
    with (
        patch.object(p.RawDrawLedger, "transactions", side_effect=AssertionError("stream touched")),
        pytest.raises(ValueError, match="working"),
    ):
        sample(request)


@pytest.mark.parametrize("kind", ["hard", "soft"])
def test_comparison_admission_precedes_sampling(kind: str) -> None:
    base = graph()
    original = base.admit()["estimated_working_bytes"]
    comparison = p.CausalInterventionRecipe(base, (p.HardIntervention(1, (0,), (0.0,)),)).admit()[
        "estimated_working_bytes"
    ]
    bounded = replace(base, limits=p.CausalLimits(working_bytes=(original + comparison) // 2))
    bounded.admit()
    request = p.InterventionPolicyRequest(
        bounded,
        (3,),
        kind=kind,
        profile="explicit-policy-r0-v1",
        eligibility=(1,),
        count=1,
        reference_rows=32,
        parameter_laws=(p.InterventionParameterLaw(1, ("matrix",)),) if kind == "soft" else (),
    )
    with (
        patch.object(p.RawDrawLedger, "transactions", side_effect=AssertionError("stream touched")),
        pytest.raises(ValueError, match="comparison working"),
    ):
        sample(request)


def test_distinct_admitted_nested_limits_preserve_original_records() -> None:
    base = replace(graph(), limits=p.CausalLimits(working_bytes=512 * 1024))
    request = p.InterventionPolicyRequest(
        base,
        (3,),
        profile="explicit-policy-r0-v1",
        eligibility=(1,),
        count=1,
        mode="matched",
        reference_rows=32,
    )
    result = sample(request)
    assert result.request.base == base
    assert result.runtime.base == base
    assert p.recipe_from_dict(asdict(result)) == result
    pair = p.generate_policy_pair(result, seed=11, task_id=13)
    assert p.compare_causal_pair(pair.before, pair.after)["mode"] == "matched"
