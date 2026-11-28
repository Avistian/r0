"""Read verified local output groups one task at a time without generating again."""

import io
import json
import zipfile
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np
import torch

from r0.modules.generation.lib.groups import verify_group
from r0.modules.priors import TrainingFeature, TrainingTask


def _tensors(archive: zipfile.ZipFile, name: str) -> dict[str, torch.Tensor]:
    with np.load(io.BytesIO(archive.read(name)), allow_pickle=False) as arrays:
        return {key: torch.tensor(arrays[key]) for key in arrays.files}


def _document(archive: zipfile.ZipFile, name: str) -> Any:
    return json.loads(archive.read(name))


def load_group(path: Path) -> Iterator[TrainingTask]:
    """Verify the receipt, then yield saved tasks on CPU in manifest order.

    This consumes locally produced runner archives, not arbitrary uploaded ZIPs.
    Only one task's arrays are materialized at a time. The saved execution record
    describes generation hardware; loading on CPU never changes that provenance.
    """
    receipt = verify_group(path)
    with zipfile.ZipFile(path) as archive:
        for task_id in receipt.task_ids:
            yield _load_task(archive, task_id)


def _load_task(archive: zipfile.ZipFile, task_id: int) -> TrainingTask:
    metadata = _document(archive, f"{task_id}/predictor.json")
    card = _document(archive, f"{task_id}/card.json")
    predictor = _tensors(archive, f"{task_id}/predictor.npz")
    truth = _tensors(archive, f"{task_id}/truth.npz")
    context = torch.tensor(metadata["context_rows"], dtype=torch.long)
    query = torch.tensor(metadata["query_rows"], dtype=torch.long)
    rows = len(context) + len(query)
    if not torch.equal(torch.sort(torch.cat((context, query))).values, torch.arange(rows)):
        raise ValueError("saved context/query indices must partition all rows")
    if card["recipe"]["task_id"] != task_id:
        raise ValueError("saved task identity disagrees with its receipt")
    inputs = torch.empty((rows, len(metadata["feature_names"])), dtype=torch.float64)
    targets = torch.empty(rows, dtype=predictor["context_y"].dtype)
    inputs[context], inputs[query] = predictor["context_x"], predictor["query_x"]
    targets[context], targets[query] = predictor["context_y"], truth["query_y"]
    masks = {"context_missing", "query_missing"} & predictor.keys()
    missing_mask = None
    if masks:
        if masks != {"context_missing", "query_missing"}:
            raise ValueError("saved missing masks require both context and query arrays")
        if any(predictor[k].dtype != torch.bool for k in masks):
            raise ValueError("saved missing masks must be boolean")
        if (
            predictor["context_missing"].shape != predictor["context_x"].shape
            or predictor["query_missing"].shape != predictor["query_x"].shape
        ):
            raise ValueError("saved missing masks must match input shapes")
        missing_mask = torch.empty_like(inputs, dtype=torch.bool)
        missing_mask[context] = predictor["context_missing"]
        missing_mask[query] = predictor["query_missing"]
    return TrainingTask(
        missing_mask=missing_mask,
        source_task_id=card["source_task_id"],
        episode_id=card["episode_id"],
        task_type=metadata["task_type"],
        recipe_json=json.dumps(
            card["recipe"], sort_keys=True, separators=(",", ":"), allow_nan=False
        ),
        mechanism_json=json.dumps(
            card["mechanism"], sort_keys=True, separators=(",", ":"), allow_nan=False
        ),
        inputs=inputs,
        targets=targets,
        scores=truth["scores"],
        probabilities=truth.get("probabilities"),
        context_indices=context,
        query_indices=query,
        classes=tuple(metadata["classes"]),
        feature_schema=tuple(
            TrainingFeature(f["kind"], tuple(f["vocabulary"]))
            for f in metadata.get("feature_schema", ())
        ),
        target_origin=metadata.get("target_origin", "adapter-v1"),
    )
