"""Empirical maps use only supplied calibration and freeze explicit tie semantics."""

import json
from dataclasses import asdict, replace
from statistics import NormalDist

import numpy as np
import pytest

from r0.modules import priors as p


def test_ties_interpolation_and_normal_scores_against_hand_values() -> None:
    calibration = np.array([0.0, 0.0, 2.0, 4.0])
    values = np.array([-1.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0])
    rank = p.fit_quantile_map(calibration, kind="rank")
    uniform = p.fit_quantile_map(calibration, kind="quantile-uniform")
    normal = p.fit_quantile_map(calibration, kind="quantile-normal")
    np.testing.assert_array_equal(rank.apply(values), [0, 0.25, 0.5, 0.625, 0.75, 0.875, 1])
    expected = np.array([0, 0.25, 0.4375, 0.625, 0.75, 0.875, 1])
    np.testing.assert_array_equal(uniform.apply(values), expected)
    oracle = np.array([NormalDist().inv_cdf(float(q)) for q in np.clip(expected, 0.1, 0.9)])
    np.testing.assert_allclose(normal.apply(values), oracle, rtol=1e-14, atol=1e-14)
    assert uniform.knots == (0.0, 2.0, 4.0)
    assert uniform.counts == (2, 1, 1)
    restored = p.fitted_quantile_map_from_dict(json.loads(json.dumps(asdict(normal))))
    np.testing.assert_array_equal(restored.apply(values), normal.apply(values))


def test_query_values_do_not_refit_or_mutate_calibration() -> None:
    context = np.array([1.0, 2.0, 3.0])
    before = context.copy()
    fitted = p.fit_quantile_map(context, kind="quantile-normal")
    frozen = asdict(fitted)
    fitted.apply(np.array([-1e100, 1e100]))
    assert asdict(fitted) == frozen
    np.testing.assert_array_equal(context, before)
    changed = p.fit_quantile_map(np.array([1.0, 2.0, 4.0]), kind="quantile-normal")
    assert changed.calibration_sha256 != fitted.calibration_sha256


def test_singleton_constants_and_float64_extremes() -> None:
    singleton = p.fit_quantile_map(np.array([2.0]), kind="quantile-normal")
    assert singleton.tail_probability == 0.25
    assert singleton.apply(np.array([2.0]))[0] == 0
    np.testing.assert_array_equal(
        p.fit_quantile_map(np.array([2.0, 2.0]), kind="quantile-uniform").apply(
            np.array([1.0, 2.0, 3.0])
        ),
        [0.0, 0.5, 1.0],
    )
    largest = np.finfo(np.float64).max
    fitted = p.fit_quantile_map(np.array([-largest, largest]), kind="quantile-uniform")
    with np.errstate(all="raise"):
        np.testing.assert_array_equal(
            fitted.apply(np.array([-largest, 0.0, largest])), [0.25, 0.5, 0.75]
        )


def test_invalid_fits_and_frozen_records_fail() -> None:
    with pytest.raises(ValueError, match="finite float64"):
        p.fit_quantile_map(np.array([np.nan]), kind="rank")
    with pytest.raises(ValueError, match="finite float64"):
        p.fit_quantile_map(np.array([], dtype=np.float64), kind="rank")
    fitted = p.fit_quantile_map(np.array([0.0, 1.0]), kind="rank")
    with pytest.raises(ValueError, match="invalid frozen"):
        replace(fitted, counts=(0, 2))
    with pytest.raises(ValueError, match="invalid frozen"):
        replace(fitted, knots=(1.0, 0.0))
    with pytest.raises(ValueError, match="invalid frozen"):
        p.fit_quantile_map(np.array([0.0, 1.0]), kind="quantile-normal", tail_probability=1e-100)
    with pytest.raises(ValueError, match="fields"):
        p.fitted_quantile_map_from_dict({**asdict(fitted), "ignored": True})
