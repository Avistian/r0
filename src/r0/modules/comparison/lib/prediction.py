"""One context-only fit, inference check and output format for every predictor."""

import time
from pathlib import Path

import numpy as np

from .contracts import Array, Predictor, validate_predictions, validate_view
from .execution import environment
from .predictors import Baseline, FitSpy
from .scoring import check_invariance
from .storage import Document, read_json, write_json


def predict(
    view: Document, predictor: str | Predictor, *, spy: FitSpy | None = None
) -> tuple[Array, Document]:
    validate_view(view)
    backend = Baseline(predictor, spy) if isinstance(predictor, str) else predictor
    context: Document = {
        "version": 1,
        "features": view["features"],
        "task": view["task"],
        "classes": view["classes"],
        "row_ids": view["context_ids"],
        "values": view["context_values"],
        "targets": view["context_targets"],
    }
    if spy and not isinstance(backend, Baseline):
        spy("backend", tuple(context["row_ids"]))
    backend.fit(context)

    def infer(indices: list[int]) -> Array:
        queries: Document = {
            "version": 1,
            "features": view["features"],
            "row_ids": [view["query_ids"][i] for i in indices],
            "values": [view["query_values"][i] for i in indices],
        }
        ids, values = backend.predict(queries)
        if len(ids) != len(set(ids)) or set(ids) != set(queries["row_ids"]):
            raise ValueError("invalid backend prediction IDs")
        values = np.asarray(values, dtype=np.float64)
        validate_predictions(view["task"], view["classes"], len(indices), values)
        return values[[ids.index(i) for i in queries["row_ids"]]]

    indices = list(range(len(view["query_ids"])))
    prediction = infer(indices)
    check_invariance(
        prediction, infer(indices[::-1])[::-1], np.concatenate([infer([i]) for i in indices])
    )
    evidence = (
        backend.evidence if isinstance(backend, Baseline) else {"fit_ids": context["row_ids"]}
    )
    return prediction, evidence


def prediction_worker(
    input_path: Path, output: Path, predictor: str | Predictor, identity: Document | None = None
) -> None:
    view, runtime = read_json(input_path), environment()
    if identity is not None and identity["environment"] != runtime:
        raise ValueError("external runtime differs from frozen environment")
    started = time.monotonic()
    prediction, evidence = predict(view, predictor)
    output.mkdir(exist_ok=False)
    np.save(output / "predictions.npy", prediction, allow_pickle=False)
    write_json(
        output / "prediction.json",
        {
            "version": 1,
            "query_ids": view["query_ids"],
            "backend_identity": identity,
            "evidence": evidence,
            "invariance": "PASS",
            "environment": runtime,
            "seconds": time.monotonic() - started,
        },
    )
