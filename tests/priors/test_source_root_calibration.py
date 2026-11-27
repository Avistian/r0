"""Frozen calibration must not normalize against query rows."""

import copy

import numpy as np
import pytest

from r0.modules import priors as p


def test_frozen_root_calibration_matches_independent_reference():
    law = p.SourceRootLaw(
        (p.SourceRootCoordinate("gaussian"), p.SourceRootCoordinate("zipf", clip=10))
    )
    rng = np.random.default_rng(77)
    oracle = copy.deepcopy(rng)
    raw = p.draw_source_root(law, rows=31, rng=oracle)
    ledger = p.RawDrawLedger()
    calibration = p.calibrate_source_root(
        law, modes=("standardize", "center"), rows=31, rng=rng, capture=ledger
    )
    query = np.array([[100.0, 10.0], [-100.0, 1.0]])
    expected = (query - raw.mean(axis=0)) / np.array([raw[:, 0].std(ddof=1), 1.0])
    np.testing.assert_allclose(calibration.apply(query), expected)
    np.testing.assert_array_equal(calibration.apply(query)[:1], calibration.apply(query[:1]))
    np.testing.assert_array_equal(rng.random(5), oracle.random(5))
    assert len(ledger.freeze()) == 2


def test_degenerate_calibration_restores_rng_and_ledger():
    law = p.SourceRootLaw((p.SourceRootCoordinate("gaussian", scale=0, location=0.1),))
    rng = np.random.default_rng(77)
    oracle = copy.deepcopy(rng)
    ledger = p.RawDrawLedger()
    with pytest.raises(ValueError, match="degenerate"):
        p.calibrate_source_root(law, modes=("standardize",), rows=37, rng=rng, capture=ledger)
    assert ledger.freeze() == ()
    np.testing.assert_array_equal(rng.random(5), oracle.random(5))


@pytest.mark.parametrize("engine", ["PCG64", "PCG64DXSM", "Philox", "SFC64", "MT19937"])
def test_calibration_raw_custody_and_centered_singleton(engine: str):
    law = p.SourceRootLaw((p.SourceRootCoordinate("unit_uniform"),))
    rng = np.random.Generator(getattr(np.random, engine)(37))
    replay = copy.deepcopy(rng)
    ledger = p.RawDrawLedger()
    result = p.calibrate_source_root(law, modes=("center",), rows=1, rng=rng, capture=ledger)
    raw = ledger.freeze()[0].replay(replay)
    assert result.locations == (float(raw[0]),)
    assert result.scales == (1.0,)
    np.testing.assert_array_equal(result.apply(raw.reshape(1, 1)), [[0.0]])
    np.testing.assert_array_equal(rng.random(5), replay.random(5))


def test_calibration_identity_and_invalid_modes_preserve_stream():
    law = p.SourceRootLaw((p.SourceRootCoordinate("gaussian", scale=0),))
    rng = np.random.default_rng(77)
    result = p.calibrate_source_root(law, modes=("identity",), rows=1, rng=rng)
    query = np.array([[1e300], [-1e300]])
    np.testing.assert_array_equal(result.apply(query), query)
    oracle = copy.deepcopy(rng)
    for modes, rows in [((), 3), (("batch",), 3), (("standardize",), 1)]:
        with pytest.raises(ValueError, match="root"):
            p.calibrate_source_root(law, modes=modes, rows=rows, rng=rng)
    np.testing.assert_array_equal(rng.random(5), oracle.random(5))


def test_calibration_resource_rejection_and_invalid_transform():
    law = p.SourceRootLaw((p.SourceRootCoordinate("gaussian"),))
    rng = np.random.default_rng(77)
    oracle = copy.deepcopy(rng)
    with pytest.raises(ValueError, match="limit"):
        p.calibrate_source_root(
            law, modes=("center",), rows=31, rng=rng, limits=p.CausalLimits(working_bytes=1)
        )
    np.testing.assert_array_equal(rng.random(5), oracle.random(5))
    with pytest.raises(ValueError, match="unused"):
        p.SourceRootCalibration(law, ("identity",), 2, (1.0,), (1.0,))
    result = p.SourceRootCalibration(law, ("center",), 2, (-1e308,), (1.0,))
    with pytest.raises(ValueError, match="nonfinite"):
        result.apply(np.array([[1e308]]))
    with pytest.raises(ValueError, match="binary64"):
        result.apply(np.array([[float("nan")]]))


def test_frozen_calibration_strict_roundtrip():
    from dataclasses import asdict

    law = p.SourceRootLaw((p.SourceRootCoordinate("unit_uniform"),))
    result = p.calibrate_source_root(
        law, modes=("standardize",), rows=10, rng=np.random.default_rng(77)
    )
    encoded = asdict(result)
    assert p.source_root_calibration_from_dict(encoded) == result
    with pytest.raises(ValueError, match="fields"):
        p.source_root_calibration_from_dict({**encoded, "ignored": True})
    with pytest.raises(ValueError, match="version"):
        p.source_root_calibration_from_dict({**encoded, "version": "batch-reference-v1"})
