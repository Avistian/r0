"""Independent weight/bank arithmetic, lifecycle isolation, custody and admission."""

import copy
import json
from dataclasses import asdict, replace
from typing import Any
from unittest.mock import patch

import numpy as np
import pytest

from r0.modules import priors as p

BANK = ((1.0, 2.0), (2.0, 4.0), (5.0, 10.0))


@pytest.mark.parametrize("engine", ["PCG64", "PCG64DXSM", "Philox", "SFC64", "MT19937"])
@pytest.mark.parametrize(
    "mode",
    [
        "multinomial-trial-temperature-r0",
        "multinomial-probability-temperature-r0",
        "dirichlet-concentration-r0",
    ],
)
def test_shared_weight_arithmetic_and_row_prefix(engine: str, mode: str):
    law = p.PrototypeWeightLaw(
        mode=mode,
        trials=None if mode.startswith("dirichlet") else 7,
        beta=1.7,
        probabilities=(0.2, 0.3, 0.5),
    )
    root = p.FrozenPrototypeRoot(BANK, law)
    rng = np.random.Generator(getattr(np.random, engine)(71))
    oracle = copy.deepcopy(rng)
    ledger = p.RawDrawLedger()
    draw = p.draw_prototype_root(root, rows=17, rng=rng, capture=ledger)
    probs = np.asarray((0.2, 0.3, 0.5))
    if mode == "dirichlet-concentration-r0":
        raw = oracle.gamma(1.7 * 3 * probs, size=(17, 3))
        expected = raw / raw.sum(axis=1, keepdims=True)
        assert draw.counts is None
    else:
        trials = 12 if mode == "multinomial-trial-temperature-r0" else 7
        if mode == "multinomial-probability-temperature-r0":
            logits = np.log(probs)
            probs = np.exp((logits - logits.max()) / 1.7)
            probs = probs / probs.sum()
        raw = oracle.multinomial(trials, probs, size=17)
        expected = raw / trials
        np.testing.assert_array_equal(draw.counts, raw)
    np.testing.assert_allclose(draw.weights, expected, rtol=1e-14, atol=1e-14)
    np.testing.assert_allclose(draw.values, expected @ np.asarray(BANK), rtol=1e-14, atol=1e-14)
    np.testing.assert_array_equal(np.asarray(draw.values)[:, 1], 2 * np.asarray(draw.values)[:, 0])
    np.testing.assert_array_equal(rng.random(9), oracle.random(9))
    short = p.draw_prototype_root(
        root, rows=4, rng=np.random.Generator(getattr(np.random, engine)(71))
    )
    assert short.values == draw.values[:4]
    assert short.weights == draw.weights[:4]
    replay = np.random.Generator(getattr(np.random, engine)(71))
    np.testing.assert_array_equal(ledger.freeze()[0].replay(replay), raw)
    assert p.frozen_prototype_root_from_dict(asdict(root)) == root


@pytest.mark.parametrize(
    ("rule", "expected"), [("ceil-r0", 3), ("floor-r0", 2), ("nearest-even-r0", 2)]
)
def test_count_conventions_and_independent_bank_lifecycle(rule: str, expected: int):
    plan = p.PrototypeRootPlan(p.JointRootBase("gaussian", 2), 10, 0.25, count_rule=rule)
    oracle = np.random.default_rng(81)
    bank = p.sample_prototype_root(plan, rng=np.random.default_rng(81))
    np.testing.assert_array_equal(bank.root.prototypes, oracle.normal(size=(expected, 2)))
    assert bank.root.count == expected
    assert bank.lifecycle == "frozen-independent-prototype-bank-r0-v1"
    assert p.sampled_prototype_root_from_dict(json.loads(json.dumps(asdict(bank)))) == bank
    old = asdict(bank)
    p.draw_prototype_root(bank.root, rows=3, rng=np.random.default_rng(13))
    p.draw_prototype_root(bank.root, rows=27, rng=np.random.default_rng(13))
    assert asdict(bank) == old


def test_source_root_prototypes_and_conditional_moments():
    law = p.SourceRootLaw(
        (p.SourceRootCoordinate("zipf", exponent=1.8), p.SourceRootCoordinate("symmetric_uniform"))
    )
    plan = p.PrototypeRootPlan(p.SourceRootBase(law), 20, 0.25)
    bank = p.sample_prototype_root(plan, rng=np.random.default_rng(817))
    oracle = np.random.default_rng(817)
    expected = np.column_stack((oracle.zipf(1.8, size=5), -1 + 2 * oracle.random(5)))
    np.testing.assert_array_equal(bank.root.prototypes, expected)
    assert bank.construction_moment_order == pytest.approx(0.8)
    assert bank.root.moment_order == float("inf")
    zero = p.FrozenPrototypeRoot(((0.0, 0.0), (0.0, 0.0)))
    draw = p.draw_prototype_root(zero, rows=7, rng=np.random.default_rng(4))
    assert np.equal(draw.values, 0).all()
    assert zero.constant_channels == (True, True)


@pytest.mark.parametrize("mode", ["multinomial-trial-temperature-r0", "dirichlet-concentration-r0"])
def test_batch_reference_binds_actual_rows_and_replays(mode: str):
    weights = p.PrototypeWeightLaw(
        mode=mode, trials=None if mode.startswith("dirichlet") else 9, beta=2.0
    )
    plan = p.PrototypeRootPlan(p.JointRootBase("uniform", 2), 16, 0.25, weights)
    batch = p.sample_prototype_batch_reference(
        plan, prototype_rng=np.random.default_rng(27), weights_rng=np.random.default_rng(28)
    )
    bank = -1 + 2 * np.random.default_rng(27).random((4, 2))
    np.testing.assert_array_equal(batch.root.prototypes, bank)
    np.testing.assert_allclose(
        batch.draw.values, np.asarray(batch.draw.weights) @ bank, rtol=0, atol=0
    )
    assert p.prototype_batch_reference_from_dict(json.loads(json.dumps(asdict(batch)))) == batch
    larger = p.sample_prototype_batch_reference(
        replace(plan, reference_rows=32),
        prototype_rng=np.random.default_rng(27),
        weights_rng=np.random.default_rng(28),
    )
    assert larger.root.count == 8
    assert len(larger.draw.values) == 32
    with pytest.raises(ValueError, match="fields"):
        p.sampled_prototype_root_from_dict(asdict(batch))
    with pytest.raises(ValueError, match="frozen bank"):
        p.draw_prototype_root(batch, rows=3, rng=np.random.default_rng(1))  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "part", ["bank", "counts", "weights", "raw", "initial", "source", "lifecycle", "extra"]
)
def test_batch_tamper_rejects(part: str):
    plan = p.PrototypeRootPlan(p.JointRootBase("gaussian", 2), 16, 0.25)
    batch = p.sample_prototype_batch_reference(
        plan, prototype_rng=np.random.default_rng(1), weights_rng=np.random.default_rng(2)
    )
    data = json.loads(json.dumps(asdict(batch)))
    if part == "bank":
        data["root"]["prototypes"][0][0] += 0.25
    elif part == "counts":
        data["draw"]["counts"][0][0] += 1
    elif part == "weights":
        data["draw"]["weights"][0][0] += 0.1
    elif part == "raw":
        data["raw"][0]["values"][0] += 0.2
    elif part == "initial":
        data["initial"][0]["digest"] = "bad"
    elif part == "source":
        data["source"] = "paper-exact"
    elif part == "lifecycle":
        data["lifecycle"] = "frozen-independent-prototype-bank-r0-v1"
    else:
        data["ignored"] = True
    with pytest.raises(ValueError, match="prototype|raw|snapshot|fields|weight|count|replay"):
        p.prototype_batch_reference_from_dict(data)


@pytest.mark.parametrize("kind", ["draws", "numbers", "working", "caller", "alias", "late"])
def test_batch_failures_are_atomic_and_preflight_known_caps(kind: str):
    weights = (
        p.PrototypeWeightLaw(mode="dirichlet-concentration-r0", trials=None)
        if kind == "late"
        else p.PrototypeWeightLaw()
    )
    plan = p.PrototypeRootPlan(p.JointRootBase("gaussian", 2), 16, 0.25, weights)
    if kind == "draws":
        plan = replace(plan, raw_limits=p.RawDrawLimits(draws=1))
    if kind == "numbers":
        plan = replace(plan, raw_limits=p.RawDrawLimits(numbers=10))
    if kind == "working":
        plan = replace(plan, raw_limits=p.RawDrawLimits(working_bytes=2 * 1024**2 + 1024))
    rng1, rng2 = np.random.default_rng(21), np.random.default_rng(22)
    before = copy.deepcopy((rng1, rng2))
    limits = p.CausalLimits(working_bytes=1) if kind == "caller" else p.CausalLimits()
    if kind == "late":
        # Real gamma underflow, following successful retained bank construction.
        plan = replace(
            plan,
            weights=p.PrototypeWeightLaw(
                mode="dirichlet-concentration-r0", trials=None, beta=1e-320
            ),
        )
        with pytest.raises(ValueError, match="degenerate"):
            p.sample_prototype_batch_reference(plan, prototype_rng=rng1, weights_rng=rng2)
    else:
        with (
            patch.object(p.RawDrawLedger, "draw", side_effect=AssertionError("draw before caps")),
            pytest.raises(ValueError, match="limit|distinct"),
        ):
            p.sample_prototype_batch_reference(
                plan,
                prototype_rng=rng1,
                weights_rng=rng1 if kind == "alias" else rng2,
                limits=limits,
            )
    np.testing.assert_array_equal(rng1.random(8), before[0].random(8))
    np.testing.assert_array_equal(rng2.random(8), before[1].random(8))


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(rho=0.0),
        dict(rho=0.001, count_rule="floor-r0"),
        dict(rho=1.0, max_prototypes=3),
        dict(reference_rows=True),
    ],
)
def test_invalid_count_profiles_reject(kwargs: dict[str, Any]):
    args: dict[str, Any] = {
        "base": p.JointRootBase("gaussian", 2),
        "reference_rows": 16,
        "rho": 0.25,
        **kwargs,
    }
    with pytest.raises(ValueError, match="prototype"):
        p.PrototypeRootPlan(**args)


@pytest.mark.parametrize("beta", [0.0, float("inf"), 1e308])
def test_beta_overflow_and_invalid_beta_reject_without_clipping(beta: float):
    with pytest.raises(ValueError, match="weight|trial"):
        p.PrototypeWeightLaw(beta=beta)
