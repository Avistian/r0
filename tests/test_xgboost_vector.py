"""Literal independent routing/rounding checks for the vector-leaf representation."""

from dataclasses import replace

import numpy as np
import pytest

from r0.modules.priors.lib.xgboost_vector import (
    FrozenXGBoostVector,
    XGBoostVectorTree,
    evaluate_xgboost_vector,
)
from r0.modules.priors.lib.xgboost_vector_export import export_xgboost_vector


def _model() -> FrozenXGBoostVector:
    tree = XGBoostVectorTree(
        (0, -1, -1),
        (0.5, 0.0, 0.0),
        (1, -1, -1),
        (2, -1, -1),
        (True, False, False),
        ((0.0, 0.0), (1.0, -2.0), (-3.0, 4.0)),
    )
    return FrozenXGBoostVector(1, (0.25, -0.5), (tree,))


def test_vector_routing_uses_strict_float32_threshold_and_nan_default() -> None:
    half = np.float32(0.5)
    inputs = np.array(
        [
            [np.nextafter(half, np.float32(-np.inf))],
            [half],
            [np.nextafter(half, np.float32(np.inf))],
            [np.nan],
        ],
        dtype=np.float64,
    )
    expected = np.array([[1.25, -2.5], [-2.75, 3.5], [-2.75, 3.5], [1.25, -2.5]], dtype=np.float32)
    np.testing.assert_array_equal(evaluate_xgboost_vector(_model(), inputs), expected)


def test_vector_model_rounds_after_each_boosting_addition() -> None:
    leaf = XGBoostVectorTree((-1,), (0.0,), (-1,), (-1,), (False,), ((1.0, -1.0),))
    model = FrozenXGBoostVector(1, (16777216.0, -16777216.0), (leaf, leaf))
    np.testing.assert_array_equal(
        evaluate_xgboost_vector(model, np.zeros((1, 1))),
        np.array([[16777216.0, -16777216.0]], dtype=np.float32),
    )


@pytest.mark.parametrize("objective", ["binary:logistic", "multi:softprob"])
def test_vector_objective_link(objective: str) -> None:
    model = replace(_model(), objective=objective)
    inputs = np.zeros((1, 1))
    margin = np.array([[1.25, -2.5]], dtype=np.float64)
    expected = (
        1 / (1 + np.exp(-margin))
        if objective == "binary:logistic"
        else (np.exp(margin - margin.max()) / np.exp(margin - margin.max()).sum())
    )
    np.testing.assert_allclose(evaluate_xgboost_vector(model, inputs), expected, rtol=2e-7)


@pytest.mark.parametrize("value", [np.inf, -np.inf, 1e300])
def test_vector_rejects_float32_overflow(value: float) -> None:
    with pytest.raises(ValueError, match="overflow"):
        evaluate_xgboost_vector(_model(), np.array([[value]]))


def test_vector_admission_precedes_float_conversion() -> None:
    with pytest.raises(ValueError, match="admission"):
        evaluate_xgboost_vector(_model(), np.full((2, 1), 1e300), max_cells=1)


@pytest.mark.parametrize("raw", [b"{}", b"[]", b"null", b'{"version":[3,2,0]}'])
def test_vector_native_export_rejects_invalid_models(raw: bytes) -> None:
    with pytest.raises(ValueError, match="XGBoost"):
        export_xgboost_vector(raw)
