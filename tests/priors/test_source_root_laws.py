"""Independent channel arithmetic, raw custody and atomic root emission."""

import copy
from dataclasses import asdict

import numpy as np
import pytest

from r0.modules import priors as p


def law():
    return p.SourceRootLaw(
        (
            p.SourceRootCoordinate("gaussian", location=2.0, scale=3.0),
            p.SourceRootCoordinate("unit_uniform"),
            p.SourceRootCoordinate("symmetric_uniform"),
            p.SourceRootCoordinate("categorical", probabilities=(0.2, 0.0, 0.8)),
            p.SourceRootCoordinate("zipf", exponent=2.5, clip=10),
            p.SourceRootCoordinate("zipf", exponent=2.5),
        )
    )


@pytest.mark.parametrize("engine", ["PCG64", "PCG64DXSM", "Philox", "SFC64", "MT19937"])
def test_root_channel_equations_and_raw_replay(engine: str):
    rng = np.random.Generator(getattr(np.random, engine)(57))
    oracle = copy.deepcopy(rng)
    replay = copy.deepcopy(rng)
    ledger = p.RawDrawLedger()
    result = p.draw_source_root(law(), rows=31, rng=rng, capture=ledger)
    expected = np.column_stack(
        (
            oracle.normal(size=31) * 3 + 2,
            oracle.random(size=31),
            -1 + 2 * oracle.random(size=31),
            oracle.choice(3, size=31, p=(0.2, 0.0, 0.8)),
            np.minimum(oracle.zipf(2.5, size=31), 10),
            oracle.zipf(2.5, size=31),
        )
    )
    np.testing.assert_array_equal(result, expected)
    records = ledger.freeze()
    assert len(records) == 6
    for record in records:
        np.testing.assert_array_equal(record.replay(replay), record.array())
    assert records[4].distribution == "zipf"
    assert records[4].dtype == "<i8"
    assert p.source_root_from_dict(asdict(law())) == law()
    tail = oracle.random(7)
    np.testing.assert_array_equal(rng.random(7), tail)
    np.testing.assert_array_equal(replay.random(7), tail)


@pytest.mark.parametrize("capture", [False, True])
def test_root_late_overflow_rolls_back_with_or_without_capture(capture: bool):
    root = p.SourceRootLaw(
        (
            p.SourceRootCoordinate("unit_uniform"),
            p.SourceRootCoordinate("gaussian", scale=1e308, location=1e308),
        )
    )
    rng = np.random.default_rng(57)
    oracle = copy.deepcopy(rng)
    ledger = p.RawDrawLedger() if capture else None
    with pytest.raises(ValueError, match="nonfinite"):
        p.draw_source_root(root, rows=128, rng=rng, capture=ledger)
    np.testing.assert_array_equal(rng.random(7), oracle.random(7))
    if ledger:
        assert ledger.freeze() == ()


def test_source_root_caps_and_unused_controls_reject():
    rng = np.random.default_rng(57)
    oracle = copy.deepcopy(rng)
    with pytest.raises(ValueError, match="limit"):
        p.draw_source_root(law(), rows=31, rng=rng, limits=p.CausalLimits(working_bytes=1))
    np.testing.assert_array_equal(rng.random(7), oracle.random(7))
    with pytest.raises(ValueError, match="unused"):
        p.SourceRootCoordinate("unit_uniform", exponent=3.0)
    with pytest.raises(ValueError, match="fields"):
        p.source_root_from_dict({**asdict(law()), "ignored": 1})


def test_root_late_failure_keeps_existing_capture_and_named_ownership():
    rng = np.random.default_rng(57)
    ledger = p.RawDrawLedger()
    p.draw_source_root(law(), rows=8, rng=rng, capture=ledger, owner=("graph", "root:3"))
    original = ledger.freeze()
    assert all(record.owner[:2] == ("graph", "root:3") for record in original)
    before = copy.deepcopy(rng)
    bad = p.SourceRootLaw((p.SourceRootCoordinate("gaussian", scale=1e308, location=1e308),))
    with pytest.raises(ValueError, match="nonfinite"):
        p.draw_source_root(bad, rows=128, rng=rng, capture=ledger)
    assert ledger.freeze() == original
    np.testing.assert_array_equal(rng.random(7), before.random(7))
