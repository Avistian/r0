"""Float64 query-only scores with no probability repair."""

import numpy as np

from .predictors import Array
from .storage import Document


def score(
    task: str,
    classes: list[str],
    query_ids: list[str],
    targets: list[object],
    prediction_ids: list[str],
    predictions: Array,
) -> Document:
    if not query_ids or len(set(query_ids)) != len(query_ids) or len(targets) != len(query_ids):
        raise ValueError("invalid scoring rows")
    if len(set(prediction_ids)) != len(prediction_ids) or set(prediction_ids) != set(query_ids):
        raise ValueError("missing, extra or duplicate prediction rows")
    p = np.asarray(predictions, dtype=np.float64)
    if p.ndim == 0 or p.shape[0] != len(prediction_ids) or not np.isfinite(p).all():
        raise ValueError("nonfinite or missing predictions")
    p = p[[prediction_ids.index(i) for i in query_ids]]
    if task == "regression":
        if p.shape != (len(query_ids),):
            raise ValueError("regression requires one scalar per row")
        y = np.asarray(targets, dtype=np.float64)
        if not np.isfinite(y).all():
            raise ValueError("nonfinite targets")
        return {"rmse": float(np.sqrt(np.mean((p - y) ** 2))), "mae": float(np.mean(np.abs(p - y)))}
    if task not in ("binary", "multiclass") or len(set(classes)) != len(classes):
        raise ValueError("invalid task/classes")
    if p.shape != (len(query_ids), len(classes)):
        raise ValueError("invalid probability dimensions")
    if (
        (p < -1e-6).any()
        or (p > 1 + 1e-6).any()
        or not np.allclose(p.sum(axis=1), 1.0, atol=1e-6, rtol=0)
    ):
        raise ValueError("invalid probabilities; no clipping or renormalization")
    if any(t not in classes for t in targets):
        raise ValueError("undeclared scoring target")
    indices = np.array([classes.index(str(t)) for t in targets])
    y = np.eye(len(classes), dtype=np.float64)[indices]
    return {
        "brier": float(np.mean(np.sum((p - y) ** 2, axis=1))),
        "accuracy": float(np.count_nonzero(p.argmax(axis=1) == indices)) / len(indices),
    }


def check_invariance(expected: Array, reordered: Array, chunks: Array) -> None:
    for actual in (reordered, chunks):
        if actual.shape != expected.shape or not np.allclose(
            expected, actual, atol=1e-6, rtol=1e-5
        ):
            raise ValueError("query order/chunk invariance failed")
