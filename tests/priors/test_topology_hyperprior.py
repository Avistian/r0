"""Public size/redirection hyperprior laws, separate from fixed graph priors."""

import json
import math
from dataclasses import asdict, replace
from unittest.mock import patch

import numpy as np
import pytest

from r0.modules import priors as p


def test_explicit_hyperprior_transform_preserves_named_rounding_and_gamma_rate():
    request = p.TopologyHyperprior(
        log_nodes_low=math.log(4),
        log_nodes_high=math.log(16),
        rounding="nearest-even",
        gamma_shape=2.0,
        gamma_rate=3.0,
        components=2,
    )
    realized = p.transform_topology_hyperprior(request, log_nodes=math.log(8), gamma_unit=6.0)
    assert realized.family == "redirection"
    assert realized.nodes == 8
    assert realized.components == 2
    assert realized.redirection_probability == 2 / 3


def test_sampled_hyperprior_retains_gamma_log_and_full_graph():
    request = p.TopologyHyperprior(
        log_nodes_low=math.log(4), log_nodes_high=math.log(8), components=2
    )
    sampled = p.sample_topology_hyperprior(request, rng=np.random.default_rng(19))
    assert sampled.log_nodes is not None
    assert sampled.topology.nodes == request.node_count(sampled.log_nodes)
    assert sum(map(len, sampled.parents)) == sampled.topology.nodes - request.components
    assert sampled.replay() == sampled


@pytest.mark.parametrize(
    "engine",
    [np.random.PCG64, np.random.PCG64DXSM, np.random.Philox, np.random.SFC64, np.random.MT19937],
)
@pytest.mark.parametrize("fixed", [False, True])
@pytest.mark.parametrize("transform", ["gamma-odds-r0", "gamma-exp-r0"])
def test_five_engine_literal_draw_and_parent_oracle(
    engine: type[np.random.BitGenerator], fixed: bool, transform: str
):
    request = (
        p.FixedSizeTopologyHyperprior(
            9, gamma_shape=2.5, gamma_rate=0.7, components=2, probability_transform=transform
        )
        if fixed
        else p.TopologyHyperprior(
            log_nodes_low=math.log(5),
            log_nodes_high=math.log(11),
            gamma_shape=2.5,
            gamma_rate=0.7,
            components=2,
            probability_transform=transform,
        )
    )
    rng, oracle = np.random.Generator(engine(73)), np.random.Generator(engine(73))
    log = None if fixed else float(oracle.uniform(math.log(5), math.log(11), size=1)[0])
    n = 9 if log is None else round(math.exp(log))
    unit = float(oracle.gamma(2.5, size=1)[0])
    gamma = unit / 0.7
    probability = gamma / (1 + gamma) if transform == "gamma-odds-r0" else -math.expm1(-gamma)
    parents: list[tuple[int, ...]] = []
    spans = tuple((n * c // 2, n * (c + 1) // 2) for c in range(2))
    for start, stop in spans:
        parents.append(())
        for node in range(start + 1, stop):
            selected = start + int(oracle.integers(node - start, size=1)[0])
            redirect = float(oracle.random(size=1)[0]) < probability
            parents.append((parents[selected][0] if redirect and parents[selected] else selected,))
    result = p.sample_topology_hyperprior(request, rng=rng)
    assert result.log_nodes == log
    assert result.gamma_unit == unit
    assert result.gamma_value == gamma
    assert result.topology.nodes == n
    assert result.topology.redirection_probability == pytest.approx(probability, rel=1e-15)
    assert result.parents == tuple(parents)
    assert result.components == spans
    assert sum(map(len, result.parents)) == n - 2
    assert repr(rng.bit_generator.state) == repr(oracle.bit_generator.state)
    restored = p.sampled_topology_hyperprior_from_dict(json.loads(json.dumps(asdict(result))))
    assert restored == result
    assert restored.replay() == result
    assert len(result.raw) == (1 if fixed else 2) + 2 * (n - 2)


@pytest.mark.parametrize("rounding", ["floor", "ceil", "nearest-even"])
def test_rounding_endpoints_and_no_hidden_clipping(rounding: str):
    request = p.TopologyHyperprior(
        log_nodes_low=0.0, log_nodes_high=math.log(20), rounding=rounding
    )
    rule = {"floor": math.floor, "ceil": math.ceil, "nearest-even": round}[rounding]
    for log in (0.0, math.log(2.5), math.log(19.5), request.log_nodes_high):
        result = p.transform_topology_hyperprior(request, log_nodes=log, gamma_unit=2.0)
        assert result.nodes == rule(math.exp(log))
    with pytest.raises(ValueError, match="support"):
        p.transform_topology_hyperprior(request, log_nodes=-0.1, gamma_unit=2.0)
    with pytest.raises(ValueError, match="underflow"):
        p.transform_topology_hyperprior(request, log_nodes=1.0, gamma_unit=0.0)
    overflow = replace(request, gamma_rate=1e-300)
    with pytest.raises(ValueError, match="nonfinite"):
        p.transform_topology_hyperprior(overflow, log_nodes=1.0, gamma_unit=1e300)


@pytest.mark.parametrize(
    "cap", ["nodes", "parameters", "working_bytes", "draws", "numbers", "raw_working"]
)
def test_worst_case_caps_precede_rng_clone(cap: str):
    request = p.TopologyHyperprior(log_nodes_low=math.log(4), log_nodes_high=math.log(16))
    limits, raw = p.CausalLimits(), p.RawDrawLimits()
    if cap in ("nodes", "parameters", "working_bytes"):
        limits = replace(limits, **{cap: 1})
    else:
        raw = replace(raw, **{"working_bytes" if cap == "raw_working" else cap: 1})
    rng = np.random.default_rng(17)
    before = repr(rng.bit_generator.state)
    with (
        patch("copy.deepcopy", side_effect=AssertionError("early clone")),
        pytest.raises(ValueError, match="limit"),
    ):
        p.sample_topology_hyperprior(request, rng=rng, limits=limits, raw_limits=raw)
    assert repr(rng.bit_generator.state) == before


def test_late_failure_rolls_back_and_complete_codec_rejects_missing_fields():
    request = p.FixedSizeTopologyHyperprior(5)
    rng = np.random.default_rng(41)
    before = repr(rng.bit_generator.state)
    with (
        patch.object(
            p.SampledTopologyHyperprior,
            "validate_structure",
            side_effect=ValueError("late rejection"),
        ),
        pytest.raises(ValueError, match="late rejection"),
    ):
        p.sample_topology_hyperprior(request, rng=rng)
    assert repr(rng.bit_generator.state) == before
    result = p.sample_topology_hyperprior(request, rng=rng)
    raw = asdict(result)
    for key in raw:
        with pytest.raises(ValueError, match="field"):
            p.sampled_topology_hyperprior_from_dict({k: v for k, v in raw.items() if k != key})
    with (
        patch.object(type(result.initial), "restore", side_effect=AssertionError("early restore")),
        pytest.raises(ValueError, match="limit"),
    ):
        p.sampled_topology_hyperprior_from_dict(raw, limits=replace(result.limits, nodes=1))
    with pytest.raises(ValueError, match="construction"):
        replace(result, parents=tuple(() for _ in result.parents))
