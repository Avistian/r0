"""Adversarial review cases at public predictor and scoring boundaries."""

from typing import Any

import numpy as np
import pytest

from r0.modules.comparison import predict_adapter, score


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
        predict_adapter({**adapter_view(), **change}, backend, IDENTITY)
    assert not backend.fitted


@pytest.mark.parametrize("probabilities", [[1.2, -0.2], [0.2, 0.2]])
def test_adapter_rejects_invalid_probability_vectors(probabilities: list[float]) -> None:
    with pytest.raises(ValueError, match="probabilit"):
        predict_adapter(adapter_view(), Backend(probabilities), IDENTITY)


def test_nonfinite_computed_scores_are_rejected() -> None:
    with pytest.raises(ValueError, match="nonfinite"):
        score("regression", [], ["q"], [-1e308], ["q"], np.array([1e308]))


def test_attempt_fits_estimator_once_for_inference_checks(tmp_path: Any, monkeypatch: Any) -> None:
    import importlib
    from pathlib import Path

    Ridge: Any = importlib.import_module("sklearn.linear_model").Ridge

    from r0.modules.comparison import attempt, prepare

    fits: list[int] = []
    ridge: Any = Ridge
    original = ridge.fit

    def counted_fit(self: Any, *args: Any, **kwargs: Any) -> Any:
        fits.append(1)
        return original(self, *args, **kwargs)

    monkeypatch.setattr(Ridge, "fit", counted_fit)
    prepare(Path("docs/evidence/issue-26/fixtures/regression.json"), tmp_path / "prepared")
    result = attempt(tmp_path / "prepared", tmp_path / "attempt", "linear", 1, diagnostic=True)
    assert result["status"] == "SUCCESS"
    assert len(fits) == 1
