"""Source-law quantile oracles and complete profile-to-frozen-network custody."""

import json
import math
from dataclasses import asdict, replace
from importlib import import_module
from typing import Any

import numpy as np
import pytest
import torch

from r0.modules import priors as p

truncnorm = import_module("scipy.stats").truncnorm


def request(family: str = "dense", profile: str = "p1-paper-tnlu-v1"):
    return p.NetworkHyperpriorRequest(
        p.NetworkSamplingRequest(2, (1,), families=(family,)),
        depth=p.NetworkScalarPrior("fixed", 2.0, 2.0),
        width=p.NetworkScalarPrior("fixed", 4.0, 4.0),
        weight_scale=p.NetworkScalarPrior(profile, 0.02, 0.2),
    )


@pytest.mark.parametrize("profile", ["p1-paper-tnlu-v1", "tabicl-9530bd6-tnlu-v1"])
@pytest.mark.parametrize("engine", [np.random.PCG64, np.random.Philox, np.random.MT19937])
def test_independent_source_quantile_oracle(profile: str, engine: Any):
    rng, oracle = np.random.Generator(engine(72)), np.random.Generator(engine(72))
    mean = math.exp(oracle.uniform(math.log(0.02), math.log(0.2)))
    sigma = (
        math.exp(oracle.uniform(math.log(0.02), math.log(0.2)))
        if profile == "p1-paper-tnlu-v1"
        else mean * math.exp(oracle.uniform(math.log(0.01), 0))
    )
    u = oracle.random()
    expected = truncnorm.ppf(
        u,
        -mean / sigma,
        (math.inf if profile == "p1-paper-tnlu-v1" else (1e6 - mean) / sigma),
        loc=mean,
        scale=sigma,
    )
    result = p.sample_hyperprior_network(
        request(profile=profile),
        hyper_rng=rng,
        named_rng=np.random.default_rng(8),
        raw_rng=np.random.default_rng(9),
    )
    chosen = result.parameters[-1]
    assert chosen.law.mean == mean
    assert chosen.law.sigma == sigma
    assert chosen.quantile == u
    assert chosen.value == pytest.approx(float(expected), rel=1e-12, abs=1e-15)
    assert result.request.weight_scale == chosen.value
    np.testing.assert_equal(rng.bit_generator.state, oracle.bit_generator.state)
    result.validate_replay()


@pytest.mark.parametrize(
    "family",
    ["dense", "individual-sparse", "block-sparse", "skip", "residual", "irregular", "gated"],
)
def test_all_existing_families_consume_actual_profile_and_replay(family: str):
    result = p.sample_hyperprior_network(
        request(family),
        hyper_rng=np.random.default_rng(72),
        named_rng=np.random.default_rng(8),
        raw_rng=np.random.default_rng(9),
    )
    assert result.request.widths == (4, 4, 1)
    task = p.LinearRecipe(rows=21, context_rows=9, features=2, signal_features=2, active_features=2)
    recipe = p.NetworkRecipe(task, result)
    emitted = p.generate_task(recipe, seed=7, task_id=2**54 + 3)
    restored = p.recipe_from_dict(json.loads(emitted.recipe_json)["recipe"])
    assert restored == recipe
    assert torch.equal(p.generate_task(restored, seed=7, task_id=2**54 + 3).scores, emitted.scores)
    view = p.NetworkInputView(recipe, replace(task, features=4, active_features=1), (1,))
    assert p.generate_task(view, seed=7, task_id=5).inputs.shape == (21, 4)


def test_failed_realization_rolls_back_all_three_streams():
    req = replace(request(), depth=p.NetworkScalarPrior("uniform", 100.0, 200.0, rounded=True))
    streams = [np.random.default_rng(i) for i in range(3)]
    before = [str(r.bit_generator.state) for r in streams]
    with pytest.raises(ValueError, match="operational cap"):
        p.sample_hyperprior_network(
            req, hyper_rng=streams[0], named_rng=streams[1], raw_rng=streams[2]
        )
    assert [str(r.bit_generator.state) for r in streams] == before


@pytest.mark.parametrize("field", ["value", "mean", "profile", "raw", "effective", "final"])
def test_full_replay_rejects_profile_custody_tampering(field: str):
    result = p.sample_hyperprior_network(
        request(),
        hyper_rng=np.random.default_rng(72),
        named_rng=np.random.default_rng(8),
        raw_rng=np.random.default_rng(9),
    )
    recipe = p.NetworkRecipe(
        p.LinearRecipe(rows=16, context_rows=8, features=2, signal_features=2, active_features=2),
        result,
    )
    data = json.loads(json.dumps(asdict(recipe)))
    net = data["network"]
    if field == "value":
        net["parameters"][-1]["value"] += 0.1
    elif field == "mean":
        net["parameters"][-1]["law"]["mean"] += 0.1
    elif field == "profile":
        net["profile"]["weight_scale"]["family"] = "tabicl-9530bd6-tnlu-v1"
    elif field == "raw":
        net["hyper_raw"] = []
    elif field == "effective":
        net["request"]["weight_scale"] += 0.1
    else:
        net["hyper_final"] = net["hyper_initial"]
    with pytest.raises(ValueError, match="replay"):
        p.recipe_from_dict(data)


def test_extension_width_scope_bias_and_round_before_offset():
    req = replace(
        request(),
        depth=p.NetworkScalarPrior("fixed", 3.5, 3.5, offset=1.0, rounded=True),
        width=p.NetworkScalarPrior("choice", choices=(2.0, 3.0, 5.0, 7.0)),
        width_scope="per-hidden-block",
        bias_scale=p.NetworkScalarPrior("log-uniform", 0.01, 0.2),
    )
    rng = np.random.default_rng(17)
    oracle = np.random.default_rng(17)
    expected = tuple((2, 3, 5, 7)[int(oracle.integers(4))] for _ in range(5))
    result = p.sample_hyperprior_network(
        req, hyper_rng=rng, named_rng=np.random.default_rng(8), raw_rng=np.random.default_rng(9)
    )
    assert result.request.widths == (*expected, 1)
    assert result.parameters[0].value == 5.0
    assert result.request.bias_scale == result.parameters[-1].value
    assert 0.01 <= result.request.bias_scale < 0.2
    assert len({v.law for v in result.parameters if v.role.startswith("width:")}) == 1
    single = replace(
        req, depth=p.NetworkScalarPrior("fixed", 1.0, 1.0), depth_semantics="total-blocks"
    )
    one = p.sample_hyperprior_network(
        single,
        hyper_rng=np.random.default_rng(1),
        named_rng=np.random.default_rng(2),
        raw_rng=np.random.default_rng(3),
    )
    assert one.request.widths == (1,)
    assert not any(v.role.startswith("width:") for v in one.parameters)


def test_pinned_upper_truncation_and_explicit_source_factory():
    req = p.network_source_hyperprior(request().template, source="tabicl-9530bd6")
    assert (
        req.depth.family == req.width.family == req.weight_scale.family == "tabicl-9530bd6-tnlu-v1"
    )
    req = replace(
        request(profile="tabicl-9530bd6-tnlu-v1"),
        weight_scale=p.NetworkScalarPrior("tabicl-9530bd6-tnlu-v1", 9e5, 1.1e6),
    )
    result = p.sample_hyperprior_network(
        req,
        hyper_rng=np.random.default_rng(16),
        named_rng=np.random.default_rng(2),
        raw_rng=np.random.default_rng(3),
    )
    value = result.parameters[-1]
    mu = value.law.mean
    sd = value.law.sigma
    assert mu is not None
    assert sd is not None
    assert value.value <= 1e6
    expected = truncnorm.ppf(value.quantile, -mu / sd, (1e6 - mu) / sd, loc=mu, scale=sd)
    assert value.value == pytest.approx(float(expected), rel=1e-12)


def test_three_stream_rollback_after_scalar_realization_before_child():
    req = replace(request(), weight_scale=p.NetworkScalarPrior("uniform", -2.0, -1.0))
    streams = [np.random.default_rng(i) for i in range(3)]
    before = [str(r.bit_generator.state) for r in streams]
    with pytest.raises(ValueError, match="network request"):
        p.sample_hyperprior_network(
            req, hyper_rng=streams[0], named_rng=streams[1], raw_rng=streams[2]
        )
    assert [str(r.bit_generator.state) for r in streams] == before
    with pytest.raises(ValueError, match="distinct"):
        p.sample_hyperprior_network(
            request(), hyper_rng=streams[0], named_rng=streams[0], raw_rng=streams[2]
        )


def test_request_limits_and_unused_source_bias_rejected():
    with pytest.raises(ValueError, match="limit"):
        replace(request(), max_hidden_width=4096)
    template = p.NetworkSamplingRequest(
        2, (1,), families=("individual-sparse",), fan_in=False, mask_profile="p1-individual"
    )
    with pytest.raises(ValueError, match="unused"):
        p.NetworkHyperpriorRequest(template, bias_scale=p.NetworkScalarPrior())
    raw = asdict(request())
    with pytest.raises(ValueError, match="caller limits"):
        p.hyperprior_request_from_dict(raw, limits=p.CausalLimits(parameters=100))


def test_real_child_overflow_rolls_back_hyper_and_network_streams():
    req = replace(request(), bias_scale=p.NetworkScalarPrior("fixed", 1e308, 1e308))
    streams = [np.random.default_rng(i) for i in (72, 8, 9)]
    before = [str(r.bit_generator.state) for r in streams]
    with np.errstate(over="ignore"), pytest.raises(ValueError, match="finite"):
        p.sample_hyperprior_network(
            req, hyper_rng=streams[0], named_rng=streams[1], raw_rng=streams[2]
        )
    assert [str(r.bit_generator.state) for r in streams] == before


def test_independent_widths_with_fixed_multiclass_output():
    req = replace(
        request(),
        template=replace(request().template, widths=(3,)),
        width=p.NetworkScalarPrior("integer-uniform", 2.0, 6.0),
        width_scope="per-hidden-block",
    )
    result = p.sample_hyperprior_network(
        req,
        hyper_rng=np.random.default_rng(72),
        named_rng=np.random.default_rng(8),
        raw_rng=np.random.default_rng(9),
    )
    assert result.request.widths[-1] == 3
    assert len(result.parameters) == 4
    task = p.LinearRecipe(
        rows=19,
        context_rows=7,
        features=2,
        signal_features=2,
        active_features=2,
        task_type="multiclass",
        classes=3,
    )
    generated = p.generate_task(p.NetworkRecipe(task, result), seed=77, task_id=2**54 + 3)
    assert generated.scores.shape == (19, 3)
    assert generated.probabilities is not None
    torch.testing.assert_close(generated.probabilities.sum(1), torch.ones(19, dtype=torch.float64))


def test_profiled_source_keeps_matching_custody_through_nested_archive():
    result = p.sample_hyperprior_network(
        request(),
        hyper_rng=np.random.default_rng(72),
        named_rng=np.random.default_rng(8),
        raw_rng=np.random.default_rng(9),
    )
    task = p.LinearRecipe(rows=17, context_rows=8, features=2, signal_features=2, active_features=2)
    recipe = p.NetworkRecipe(task, result)
    matching = p.fit_network_variance_match(
        recipe, before_inputs=(0, 1), after_inputs=(1,), rows=29, rng=np.random.default_rng(51)
    )
    view = p.MatchedNetworkInputView(
        recipe, replace(task, active_features=1), (1,), matching=matching
    )
    restored = p.recipe_from_dict(asdict(view))
    assert restored == view
    assert torch.equal(
        p.generate_task(view, seed=7, task_id=3).scores,
        p.generate_task(restored, seed=7, task_id=3).scores,
    )
