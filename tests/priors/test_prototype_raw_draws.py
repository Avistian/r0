"""Integer multinomial custody and vector-Gamma prototype weight primitives."""

import copy
from dataclasses import replace

import numpy as np
import pytest

from r0.modules import priors as p


@pytest.mark.parametrize("engine", ["PCG64", "PCG64DXSM", "Philox", "SFC64", "MT19937"])
def test_prototype_weight_raw_laws_replay(engine: str):
    rng = np.random.Generator(getattr(np.random, engine)(904))
    oracle, replay = copy.deepcopy(rng), copy.deepcopy(rng)
    ledger = p.RawDrawLedger()
    with ledger.transaction(rng) as (stream, staged):
        counts = staged.draw(
            stream,
            owner=("prototype",),
            role="counts",
            distribution="multinomial",
            shape=(13, 3),
            parameters=(("trials", 17), ("probabilities", (0.2, 0.0, 0.8))),
        )
        gamma = staged.draw(
            stream,
            owner=("prototype",),
            role="gamma",
            distribution="gamma_vector",
            shape=(13, 3),
            parameters=(("concentrations", (0.2, 1.0, 3.0)),),
        )
    np.testing.assert_array_equal(counts, oracle.multinomial(17, (0.2, 0.0, 0.8), size=13))
    np.testing.assert_array_equal(gamma, oracle.gamma((0.2, 1.0, 3.0), size=(13, 3)))
    assert counts.dtype == np.int64
    assert np.all(counts.sum(axis=1) == 17)
    for record in ledger.freeze():
        np.testing.assert_array_equal(record.replay(replay), record.array())
    tail = oracle.random(7)
    np.testing.assert_array_equal(rng.random(7), tail)
    np.testing.assert_array_equal(replay.random(7), tail)


def test_multinomial_retained_counts_reject_corruption():
    rng = np.random.default_rng(2)
    ledger = p.RawDrawLedger()
    with ledger.transaction(rng) as (stream, staged):
        staged.draw(
            stream,
            owner=("p",),
            role="counts",
            distribution="multinomial",
            shape=(2, 3),
            parameters=(("trials", 7), ("probabilities", (0.2, 0.3, 0.5))),
        )
    record = ledger.freeze()[0]
    for value in (-1, True, 1.5, 2**63):
        with pytest.raises(ValueError, match="integer|count|dtype|numeric"):
            replace(record, values=(value, *record.values[1:]))
    with pytest.raises(ValueError, match="sum|count"):
        replace(record, values=(record.values[0] + 1, *record.values[1:]))
