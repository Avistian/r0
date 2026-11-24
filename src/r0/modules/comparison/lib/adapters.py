"""Learner/backend interface for provisioned models; fixture validation needs no checkpoint."""

from collections.abc import Callable
from typing import Protocol

import numpy as np

from .contracts import validate_predictions, validate_view
from .predictors import Array
from .storage import Document, digest


class Predictor(Protocol):
    """A backend receives a context-only fit view and a separate query view."""

    def fit(self, context: Document) -> None: ...

    def predict(self, queries: Document) -> tuple[list[str], Array]: ...


def predict_adapter(
    view: Document,
    backend: Predictor,
    identity: Document,
    *,
    spy: Callable[[str, tuple[str, ...]], None] | None = None,
) -> Document:
    """Use the versioned contract without requiring a trained checkpoint in fixture tests.

    Model loading belongs inside the worker's enforced attempt boundary. The caller
    must bind checkpoint, implementation, environment and settings in identity.
    """
    validate_view(view)
    required = {"implementation", "environment", "settings", "checkpoint", "evidence"}
    if set(identity) != required or not all(identity[k] for k in required):
        raise ValueError("incomplete adapter identity")
    context: Document = {
        "version": 1,
        "features": view["features"],
        "task": view["task"],
        "classes": view["classes"],
        "row_ids": view["context_ids"],
        "values": view["context_values"],
        "targets": view["context_targets"],
    }
    queries: Document = {
        "version": 1,
        "features": view["features"],
        "row_ids": view["query_ids"],
        "values": view["query_values"],
    }
    if spy:
        spy("backend", tuple(context["row_ids"]))
    backend.fit(context)
    ids, values = backend.predict(queries)
    if len(ids) != len(set(ids)) or set(ids) != set(view["query_ids"]):
        raise ValueError("invalid backend prediction IDs")
    values = np.asarray(values, dtype=np.float64)
    validate_predictions(view["task"], view["classes"], len(ids), values)
    return {
        "version": 1,
        "identity": identity,
        "identity_sha256": digest(identity),
        "query_ids": ids,
        "predictions": values,
        "fit_ids": context["row_ids"],
    }
