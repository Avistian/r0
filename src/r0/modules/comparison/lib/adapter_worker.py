"""Common fitted-state checks and output format for separately executed backends."""

import time
from pathlib import Path

import numpy as np

from .adapters import Predictor, predict_adapter
from .contracts import validate_predictions
from .execution import environment
from .scoring import check_invariance
from .storage import Document, read_json, write_json


def adapter_worker(input_path: Path, output: Path, backend: Predictor, identity: Document) -> None:
    view = read_json(input_path)
    runtime = environment()
    if identity["environment"] != runtime:
        raise ValueError("external runtime differs from frozen environment")
    started = time.monotonic()
    result = predict_adapter(view, backend, identity)
    ids = view["query_ids"]
    prediction = result["predictions"][[result["query_ids"].index(i) for i in ids]]

    def infer(indices: list[int]) -> np.ndarray[tuple[int, ...], np.dtype[np.float64]]:
        queries: Document = {
            "version": 1,
            "features": view["features"],
            "row_ids": [ids[i] for i in indices],
            "values": [view["query_values"][i] for i in indices],
        }
        returned_ids, values = backend.predict(queries)
        if len(set(returned_ids)) != len(indices) or set(returned_ids) != set(queries["row_ids"]):
            raise ValueError("invalid backend prediction IDs")
        values = np.asarray(values, dtype=np.float64)
        validate_predictions(view["task"], view["classes"], len(indices), values)
        return values[[returned_ids.index(i) for i in queries["row_ids"]]]

    reverse = list(reversed(range(len(ids))))
    check_invariance(
        prediction, infer(reverse)[reverse], np.concatenate([infer([i]) for i in range(len(ids))])
    )
    output.mkdir(exist_ok=False)
    np.save(output / "predictions.npy", prediction, allow_pickle=False)
    write_json(
        output / "prediction.json",
        {
            "version": 1,
            "query_ids": ids,
            "backend_identity": identity,
            "evidence": {"fit_ids": result["fit_ids"], "recipe": identity["settings"]},
            "invariance": "PASS",
            "seconds": time.monotonic() - started,
            "environment": runtime,
        },
    )
