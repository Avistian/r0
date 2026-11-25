"""Lossless arrays and readable tables with predictor and diagnostic truth separation."""

import csv
import io
import json
import zipfile
from typing import Any

import numpy as np
import torch

from r0.modules.priors import TrainingTask


def _arrays(**values: torch.Tensor) -> bytes:
    stream = io.BytesIO()
    np.savez(
        stream,
        allow_pickle=False,
        **{name: value.detach().cpu().numpy() for name, value in values.items()},
    )
    return stream.getvalue()


def replay_bundle(task: TrainingTask, *, max_bytes: int = 256 * 1024**2) -> bytes:
    """Export all rows. Query labels and mechanism information stay outside predictors."""
    if type(max_bytes) is not int or max_bytes < 65536:
        raise ValueError("max_bytes must be at least 64 KiB")
    buffer = io.BytesIO()
    context, query = task.context_indices, task.query_indices

    def check_size() -> None:
        if buffer.tell() > max_bytes:
            raise ValueError("replay bundle exceeds output-storage limit")

    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_STORED) as bundle:
        documents = {
            "recipe.json": task.recipe_json,
            "mechanism.json": task.mechanism_json,
            "identity.json": json.dumps(
                {
                    "source_task_id": task.source_task_id,
                    "episode_id": task.episode_id,
                    "classes": task.classes,
                    "task_type": task.task_type,
                }
            ),
        }
        for name, data in documents.items():
            bundle.writestr(name, data)
            check_size()
        bundle.writestr(
            "predictor.npz",
            _arrays(
                context_x=task.inputs[context],
                context_y=task.targets[context],
                query_x=task.inputs[query],
                context_rows=context,
                query_rows=query,
            ),
        )
        check_size()
        truth = {"targets": task.targets, "scores": task.scores}
        if task.probabilities is not None:
            truth["probabilities"] = task.probabilities
        bundle.writestr("truth.npz", _arrays(**truth))
        check_size()
        x = np.asarray(task.inputs.detach().cpu().numpy(), dtype=np.float64)
        y = np.asarray(task.targets.detach().cpu().numpy(), dtype=np.float64)
        scores = np.asarray(task.scores.detach().cpu().numpy(), dtype=np.float64)
        probabilities = (
            None
            if task.probabilities is None
            else np.asarray(task.probabilities.detach().cpu().numpy(), dtype=np.float64)
        )
        context_set = set(np.asarray(context.detach().cpu().numpy(), dtype=np.int64).tolist())
        for name in ("predictor.csv", "truth.csv"):
            with (
                bundle.open(name, "w") as entry,
                io.TextIOWrapper(entry, encoding="utf-8", newline="", write_through=True) as text,
            ):
                writer = csv.writer(text)
                header = (
                    ["row", "role", *[f"x{i}" for i in range(x.shape[1])], "context_target"]
                    if name == "predictor.csv"
                    else [
                        "row",
                        "target",
                        *[f"score_{c}" for c in range(scores.shape[1])],
                        *[f"probability_{c}" for c in range(len(task.classes))],
                    ]
                )
                writer.writerow(header)
                for i in range(len(x)):
                    target: int | float = int(y[i]) if task.classes else float(y[i])
                    row: list[Any] = (
                        [
                            i,
                            "context" if i in context_set else "query",
                            *x[i].tolist(),
                            target if i in context_set else "",
                        ]
                        if name == "predictor.csv"
                        else [
                            i,
                            target,
                            *scores[i].tolist(),
                            *([] if probabilities is None else probabilities[i].tolist()),
                        ]
                    )
                    writer.writerow(row)
                    if i % 256 == 0:
                        check_size()
            check_size()
    check_size()
    return buffer.getvalue()
