"""Zipf records retain integer samples and reject before publishing RNG changes."""

import copy
from dataclasses import replace

import numpy as np
import pytest

from r0.modules import priors as p


@pytest.mark.parametrize("engine", ["PCG64", "PCG64DXSM", "Philox", "SFC64", "MT19937"])
@pytest.mark.parametrize("exponent", [2.0, 3.25, 20.0])
def test_zipf_draw_record_is_lossless_and_replays(engine: str, exponent: float):
    rng = np.random.Generator(getattr(np.random, engine)(91))
    oracle = copy.deepcopy(rng)
    replay = copy.deepcopy(rng)
    ledger = p.RawDrawLedger()
    with ledger.transaction(rng) as (stream, staged):
        actual = staged.draw(
            stream,
            owner=("root",),
            role="zipf",
            distribution="zipf",
            shape=(17, 3),
            parameters=(("exponent", exponent),),
        )
    expected = oracle.zipf(exponent, size=(17, 3))
    np.testing.assert_array_equal(actual, expected)
    (record,) = ledger.freeze()
    assert record.dtype == "<i8"
    assert all(type(v) is int and v >= 1 for v in record.values)
    np.testing.assert_array_equal(record.replay(replay), expected)
    next_draws = oracle.random(5)
    np.testing.assert_array_equal(rng.random(5), next_draws)
    np.testing.assert_array_equal(replay.random(5), next_draws)
    with pytest.raises(ValueError, match="Zipf"):
        replace(record, values=(0, *record.values[1:]))


@pytest.mark.parametrize("exponent", [1.0, 0.0, -2.0, float("inf"), float("nan"), True])
def test_zipf_bad_exponent_rolls_back(exponent: float):
    rng = np.random.default_rng(4)
    before = copy.deepcopy(rng)
    ledger = p.RawDrawLedger()
    with (
        pytest.raises(ValueError, match="Zipf exponent"),
        ledger.transaction(rng) as (stream, staged),
    ):
        staged.draw(
            stream,
            owner=("root",),
            role="zipf",
            distribution="zipf",
            shape=(2,),
            parameters=(("exponent", exponent),),
        )
    assert ledger.freeze() == ()
    np.testing.assert_array_equal(rng.random(10), before.random(10))
