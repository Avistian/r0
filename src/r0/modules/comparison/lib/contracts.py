"""Shared input and output validation for every predictor backend."""

from typing import Protocol, cast

import numpy as np
from numpy.typing import NDArray

from .storage import Document


def validate_view(view: Document) -> None:
    """Reject malformed or leaking input before any fitted operation."""
    allowed = {
        "version",
        "evidence",
        "task",
        "classes",
        "features",
        "context_ids",
        "query_ids",
        "context_values",
        "query_values",
        "context_targets",
        "support",
        "limits",
    }
    if set(view) != allowed or view["version"] != 1:
        raise ValueError("invalid predictor input contract; scoring targets are forbidden")
    if view["evidence"] != "AUTHORED_FIXTURE":
        raise ValueError("real predictor execution requires an admitted frozen protocol")
    ids = view["context_ids"]
    query_ids = view["query_ids"]
    if (
        len(set(ids)) != len(ids)
        or len(set(query_ids)) != len(query_ids)
        or set(ids) & set(query_ids)
    ):
        raise ValueError("duplicate or overlapping rows")
    if (
        len(ids) != view["support"]
        or len(ids) != len(view["context_targets"])
        or len(ids) != len(view["context_values"])
        or len(query_ids) != len(view["query_values"])
    ):
        raise ValueError("support mismatch")
    if not ids or not query_ids:
        raise ValueError("empty predictor rows")
    width = len(view["features"])
    if not width or any(len(row) != width for row in view["context_values"] + view["query_values"]):
        raise ValueError("feature width mismatch")


def validate_predictions(
    task: str, classes: list[str], count: int, values: NDArray[np.float64]
) -> None:
    expected = (count,) if task == "regression" else (count, len(classes))
    if values.shape != expected or not np.isfinite(values).all():
        raise ValueError("invalid or nonfinite prediction dimensions")
    if task != "regression" and (
        (values < -1e-6).any()
        or (values > 1 + 1e-6).any()
        or not np.allclose(values.sum(axis=1), 1.0, atol=1e-6, rtol=0)
    ):
        raise ValueError("invalid probabilities; no clipping or renormalization")


Array = NDArray[np.float64]


class Predictor(Protocol):
    """Every backend fits labeled context and predicts a separate query view."""

    def fit(self, context: Document) -> None: ...

    def predict(self, queries: Document) -> tuple[list[str], Array]: ...


def validate_output(output: Document, view: Document, identity: Document | None) -> None:
    raw = output.get("query_ids")
    if not isinstance(raw, list) or not all(isinstance(i, str) for i in cast(list[object], raw)):
        raise ValueError("prediction query IDs missing or malformed")
    ids = cast(list[str], raw)
    if len(ids) != len(set(ids)) or set(ids) != set(view["query_ids"]):
        raise ValueError("prediction query IDs differ from prepared input")
    if output.get("version") != 1 or output.get("invariance") != "PASS":
        raise ValueError("unsupported prediction output or invariance not verified")
    if (
        not isinstance(output.get("evidence"), dict)
        or output["evidence"].get("fit_ids") != view["context_ids"]
    ):
        raise ValueError("prediction fit IDs differ from context")
    if output.get("backend_identity") != identity:
        raise ValueError("external backend identity mismatch")
    if identity is not None and output.get("environment") != identity["environment"]:
        raise ValueError("external runtime differs from frozen environment")
