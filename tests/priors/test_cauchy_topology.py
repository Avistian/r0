"""Cauchy topology retains full proposals and matches a separate scalar oracle."""

import copy
import math
from dataclasses import replace

import numpy as np
import pytest

from r0.modules import priors


def test_cauchy_topology_matches_scalar_source_equations_and_generator_tail() -> None:
    rng, oracle = np.random.default_rng(42), np.random.default_rng(42)
    prior = priors.CauchyGraphPrior(nodes=5, offset=-1.0)
    result = priors.sample_cauchy_graph(prior, rng=rng)
    offset = float(oracle.standard_cauchy()) - 1.0
    receiver = oracle.standard_cauchy(5)
    source = oracle.standard_cauchy(5)
    proposals = oracle.random((5, 5))
    expected: list[tuple[int, ...]] = []
    for target in range(5):
        incoming: list[int] = []
        for parent in range(target):
            z = offset + float(receiver[target]) + float(source[parent])
            p = 1 / (1 + math.exp(-z)) if z >= 0 else math.exp(z) / (1 + math.exp(z))
            if proposals[parent, target] <= p:
                incoming.append(parent)
        expected.append(tuple(incoming))
    assert result.parents == tuple(expected)
    assert len(result.raw) == 4
    assert result.raw[-1].shape == (5, 5)
    np.testing.assert_array_equal(result.raw[-1].array(), proposals)
    np.testing.assert_array_equal(rng.random(8), oracle.random(8))
    result.validate()


@pytest.mark.parametrize("engine", ["PCG64", "PCG64DXSM", "Philox", "SFC64", "MT19937"])
def test_cauchy_raw_replay_and_discarded_proposal_binding(engine: str) -> None:
    rng = np.random.Generator(getattr(np.random, engine)(42))
    before = copy.deepcopy(rng)
    result = priors.sample_cauchy_graph(priors.CauchyGraphPrior(nodes=5, offset=-1.0), rng=rng)
    control = copy.deepcopy(before)
    assert result.replay(control) == result
    np.testing.assert_array_equal(control.random(8), rng.random(8))
    # A diagonal proposal is unused for edges but remains part of the raw proof.
    proposal = result.raw[-1]
    changed = replace(proposal, values=(0.125, *proposal.values[1:]))
    corrupted = replace(result, raw=(*result.raw[:-1], changed))
    untouched = copy.deepcopy(before)
    with pytest.raises(ValueError, match="replay differs"):
        corrupted.replay(before)
    np.testing.assert_array_equal(before.random(8), untouched.random(8))


def test_cauchy_topology_rejects_wrong_types_and_out_of_range_uniforms() -> None:
    result = priors.sample_cauchy_graph(
        priors.CauchyGraphPrior(nodes=3, offset=1000.0), rng=np.random.default_rng(42)
    )
    assert result.parents == ((), (0,), (0, 1))
    with pytest.raises(ValueError, match="parent"):
        replace(result, parents=((), (False,), (0, 1))).validate()
    proposal = result.raw[-1]
    with pytest.raises(ValueError, match="proposal"):
        replace(
            result, raw=(*result.raw[:-1], replace(proposal, values=(-1.0, *proposal.values[1:])))
        ).validate()


def test_cauchy_budget_rejection_keeps_injected_state_and_extreme_draws() -> None:
    rng, control = np.random.default_rng(42), np.random.default_rng(42)
    with pytest.raises(ValueError, match="parameter limit"):
        priors.sample_cauchy_graph(
            priors.CauchyGraphPrior(nodes=100, offset=0.0),
            rng=rng,
            limits=priors.CausalLimits(parameters=1000),
        )
    np.testing.assert_array_equal(rng.random(8), control.random(8))
    result = priors.sample_cauchy_graph(
        priors.CauchyGraphPrior(nodes=5, offset=-1000.0), rng=np.random.default_rng(42)
    )
    assert result.parents == ((), (), (), (), ())
