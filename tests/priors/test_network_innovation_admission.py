"""Caller statistics streams are untouched by known compiled graph cap failures."""

import copy
from dataclasses import replace

import numpy as np
import pytest

from r0.modules import priors as p


def base(rows: int = 16) -> p.NetworkRecipe:
    return p.NetworkRecipe(
        p.LinearRecipe(rows=rows, context_rows=8, features=1, signal_features=1, active_features=1),
        p.FrozenNetworkFunction(1, (p.NetworkBlock((0,), (p.NetworkAffine(((1.0,),), (0.1,)),)),)),
    )


@pytest.mark.parametrize("cap", ["channels", "workspace", "parameters"])
def test_compiled_limits_precede_any_statistics_draw(cap: str, monkeypatch: pytest.MonkeyPatch):
    source = base(16000 if cap == "workspace" else 16)
    if cap == "channels":
        limits = replace(source.limits, total_channels=3)
    elif cap == "parameters":
        limits = replace(source.limits, parameters=2000)
    else:
        plan = p.NetworkInnovationPlan(
            (p.NetworkInnovationStage((0.0,), (1.0,), "pre-activation"),)
        )
        compiled = p.with_network_innovations(source, plan)
        old = source.runtime.admit()["estimated_working_bytes"]
        new = compiled.admit()["estimated_working_bytes"]
        assert new > old
        limits = replace(source.limits, working_bytes=(old + new) // 2)
    source = replace(source, limits=limits)
    request = p.NetworkInnovationRequest(
        p.NetworkScalarPrior("uniform", 0.1, 0.3), placement="pre-activation"
    )
    if cap == "parameters":
        choices = p.NetworkScalarPrior("choice", choices=tuple(0.1 * (i + 1) for i in range(256)))
        request = replace(request, scale=choices, mean=choices, importance=choices)
    rng = (
        np.random.Generator(np.random.MT19937(51))
        if cap == "parameters"
        else np.random.default_rng(51)
    )
    initial = copy.deepcopy(rng.bit_generator.state)

    def forbidden(*args: object, **kwargs: object):
        raise AssertionError("statistics draw occurred before compiled aggregate admission")

    monkeypatch.setattr(p.RawDrawLedger, "draw", forbidden)
    with pytest.raises(ValueError, match="channel|working|admission|parameter"):
        p.sample_network_innovations(source, request, statistics_rng=rng)
    np.testing.assert_equal(rng.bit_generator.state, initial)
