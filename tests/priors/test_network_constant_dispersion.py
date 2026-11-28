"""Constant decimal scores must not acquire variance through mean roundoff."""

import json
from dataclasses import asdict, replace

import numpy as np
import pytest

from r0.modules import priors as p


def source(second_weight: float = 0.0) -> p.NetworkRecipe:
    return p.NetworkRecipe(
        p.LinearRecipe(rows=21, context_rows=9, features=2, signal_features=2, active_features=2),
        p.FrozenNetworkFunction(
            2, (p.NetworkBlock((0,), (p.NetworkAffine(((2.0, second_weight),), (0.1,)),)),)
        ),
    )


def test_decimal_constant_rejects_without_advancing_rng():
    rng = np.random.default_rng(42)
    before = repr(rng.bit_generator.state)
    with pytest.raises(ValueError, match="constant"):
        p.fit_network_variance_match(
            source(), before_inputs=(0, 1), after_inputs=(1,), rows=37, rng=rng
        )
    assert repr(rng.bit_generator.state) == before


def test_decoder_rejects_internally_consistent_roundoff_dispersion():
    original = source(1.0)
    fitted = p.fit_network_variance_match(
        original, before_inputs=(0, 1), after_inputs=(1,), rows=37, rng=np.random.default_rng(42)
    )
    view = p.MatchedNetworkInputView(
        original, replace(original.task, active_features=1), (1,), matching=fitted
    )
    record = json.loads(json.dumps(asdict(view)))
    badbase = json.loads(json.dumps(asdict(source())))
    record["base"] = badbase
    match = record["matching"]
    match["base"] = badbase
    x = np.asarray(match["population"])
    before = 2 * x[:, :1] + 0.1
    after = np.full((37, 1), 0.1)
    assert after.std(0)[0] > 0  # This fixture exposes ordinary NumPy mean roundoff.
    for name, values in [("before", before), ("after", after)]:
        match[f"{name}_scores"] = values.tolist()
        match[f"{name}_mean"] = values.mean(0).tolist()
        match[f"{name}_std"] = values.std(0).tolist()
    match["digest"] = p.record_fingerprint(
        (match["population"], match["before_scores"], match["after_scores"])
    )
    with pytest.raises(ValueError, match="constant"):
        p.recipe_from_dict(record)


def test_tiny_nonconstant_scores_are_not_thresholded_away():
    fitted = p.fit_network_variance_match(
        source(1e-14),
        before_inputs=(0, 1),
        after_inputs=(1,),
        rows=37,
        rng=np.random.default_rng(42),
    )
    assert fitted.after_std[0] > 0
    assert fitted.after_std[0] < 1e-13
    assert len(set(fitted.after_scores)) > 1
