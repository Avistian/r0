"""Adversarial review cases at public predictor and scoring boundaries."""

from typing import Any

import numpy as np
import pytest

from r0.modules.comparison import predict, score


def adapter_view() -> dict[str, Any]:
    return {
        "version": 1,
        "evidence": "AUTHORED_FIXTURE",
        "task": "binary",
        "classes": ["a", "b"],
        "features": [{"name": "x", "kind": "numeric"}],
        "context_ids": ["c0", "c1"],
        "query_ids": ["q"],
        "context_values": [[0], [1]],
        "query_values": [[2]],
        "context_targets": ["a", "b"],
        "support": 2,
        "limits": {},
    }


IDENTITY = {
    "implementation": "review-fixture-v1",
    "environment": "fixture",
    "settings": {"fixed": True},
    "checkpoint": "FIXTURE_NO_CHECKPOINT",
    "evidence": "FIXTURE_ONLY",
}


class Backend:
    def __init__(self, probabilities: list[float]) -> None:
        self.probabilities = probabilities
        self.fitted = False

    def fit(self, context: dict[str, Any]) -> None:
        self.fitted = True

    def predict(self, queries: dict[str, Any]) -> tuple[list[str], Any]:
        return queries["row_ids"], np.array([self.probabilities])


@pytest.mark.parametrize(
    "change",
    [
        {"context_ids": ["q", "c1"]},
        {"context_ids": ["c0", "c0"]},
        {"support": 3},
        {"context_values": [[0]]},
        {"context_values": [[0, 1], [1, 2]]},
        {"query_targets": ["a"]},
    ],
)
def test_adapter_rejects_invalid_views_before_fit(change: dict[str, Any]) -> None:
    backend = Backend([0.5, 0.5])
    with pytest.raises(ValueError, match="overlapping|support|width|scoring"):
        predict({**adapter_view(), **change}, backend)
    assert not backend.fitted


@pytest.mark.parametrize("probabilities", [[1.2, -0.2], [0.2, 0.2]])
def test_adapter_rejects_invalid_probability_vectors(probabilities: list[float]) -> None:
    with pytest.raises(ValueError, match="probabilit"):
        predict(adapter_view(), Backend(probabilities))


def test_nonfinite_computed_scores_are_rejected() -> None:
    with pytest.raises(ValueError, match="nonfinite"):
        score("regression", [], ["q"], [-1e308], ["q"], np.array([1e308]))


def test_predict_fits_once_for_inference_checks() -> None:
    class Once(Backend):
        def fit(self, context: dict[str, Any]) -> None:
            assert not self.fitted
            super().fit(context)

    backend = Once([0.5, 0.5])
    predict(adapter_view(), backend)
    assert backend.fitted
