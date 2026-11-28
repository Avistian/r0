"""Source batch normalization uses this emission, not an independent fit."""

import copy
from dataclasses import asdict

import numpy as np
import pytest

from r0.modules import priors as p


@pytest.mark.parametrize("engine", ["PCG64", "PCG64DXSM", "Philox", "SFC64", "MT19937"])
def test_batch_reference_exact_equations_and_custody(engine: str):
    law = p.SourceRootLaw(
        (
            p.SourceRootCoordinate("categorical", probabilities=(0.2, 0.3, 0.5)),
            p.SourceRootCoordinate("zipf", exponent=2.3, clip=10),
        )
    )
    rng = np.random.Generator(getattr(np.random, engine)(72))
    oracle = copy.deepcopy(rng)
    replay = copy.deepcopy(rng)
    raw = p.draw_source_root(law, rows=43, rng=oracle)
    ledger = p.RawDrawLedger()
    values, record = p.draw_source_root_batch_reference(
        law, modes=("standardize", "center"), rows=43, rng=rng, capture=ledger
    )
    expected = (raw - raw.mean(axis=0)) / np.array([raw[:, 0].std(ddof=1), 1.0])
    np.testing.assert_array_equal(values, expected)
    assert record.version == "source-root-batch-reference-v1"
    assert p.source_root_batch_reference_from_dict(asdict(record)) == record
    with pytest.raises(ValueError, match="version"):
        p.source_root_calibration_from_dict(asdict(record))
    for draw in ledger.freeze():
        np.testing.assert_array_equal(draw.replay(replay), draw.array())
    tail = oracle.random(5)
    np.testing.assert_array_equal(rng.random(5), tail)
    np.testing.assert_array_equal(replay.random(5), tail)


def test_batch_degenerate_rejection_is_atomic():
    law = p.SourceRootLaw((p.SourceRootCoordinate("categorical", probabilities=(1.0, 0.0)),))
    rng = np.random.default_rng(72)
    oracle = copy.deepcopy(rng)
    ledger = p.RawDrawLedger()
    with pytest.raises(ValueError, match="degenerate"):
        p.draw_source_root_batch_reference(
            law, modes=("standardize",), rows=43, rng=rng, capture=ledger
        )
    assert ledger.freeze() == ()
    np.testing.assert_array_equal(rng.random(5), oracle.random(5))


def test_batch_membership_changes_normalization_explicitly():
    law = p.SourceRootLaw((p.SourceRootCoordinate("unit_uniform"),))
    small, before = p.draw_source_root_batch_reference(
        law, modes=("center",), rows=5, rng=np.random.default_rng(72)
    )
    large, after = p.draw_source_root_batch_reference(
        law, modes=("center",), rows=6, rng=np.random.default_rng(72)
    )
    assert before.locations != after.locations
    assert not np.array_equal(small, large[:5])
    np.testing.assert_allclose(small[:, 0] + before.locations[0], large[:5, 0] + after.locations[0])
    with pytest.raises(ValueError, match="fields"):
        p.source_root_batch_reference_from_dict({**asdict(before), "ignored": True})
    with pytest.raises(ValueError, match="version"):
        p.source_root_batch_reference_from_dict(
            {**asdict(before), "version": "source-root-independent-calibration-v1"}
        )
