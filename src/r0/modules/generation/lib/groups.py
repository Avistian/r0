"""Write-once task archives; a separate receipt marks verified completion."""

import hashlib
import io
import json
import os
import tempfile
import zipfile
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch

from r0.modules.priors import TrainingTask


class GroupLimitError(ValueError):
    """The archive cannot fit within its declared byte limit."""


class EmptyGroupError(ValueError):
    """No successful tasks were supplied for an archive."""


@dataclass(frozen=True, slots=True)
class GroupReceipt:
    sha256: str
    size_bytes: int
    task_ids: tuple[int, ...]
    version: str = "task-group-v1"


def _json(value: object) -> bytes:
    return (json.dumps(value, sort_keys=True, allow_nan=False) + "\n").encode()


def write_record(path: Path, value: object) -> None:
    """Publish a complete record atomically; retain interrupted temporary records."""
    if path.exists():
        raise FileExistsError(path)
    with tempfile.NamedTemporaryFile(
        mode="wb", dir=path.parent, prefix=f"{path.name}.", suffix=".pending", delete=False
    ) as handle:
        handle.write(_json(value))
        handle.flush()
        os.fsync(handle.fileno())
        temporary = Path(handle.name)
    os.replace(temporary, path)


def _arrays(**tensors: torch.Tensor) -> bytes:
    buffer = io.BytesIO()
    np.savez(
        buffer, allow_pickle=False, **{k: v.detach().cpu().numpy() for k, v in tensors.items()}
    )
    return buffer.getvalue()


def _entries(task: TrainingTask) -> Iterable[tuple[str, bytes]]:
    context, queries = task.context_indices, task.query_indices
    yield (
        "predictor.npz",
        _arrays(
            context_x=task.inputs[context],
            context_y=task.targets[context],
            query_x=task.inputs[queries],
            **(
                {
                    "context_missing": task.missing_mask[context],
                    "query_missing": task.missing_mask[queries],
                }
                if task.missing_mask is not None
                else {}
            ),
        ),
    )
    yield (
        "predictor.json",
        _json(
            {
                **(
                    {"feature_schema": [asdict(f) for f in task.feature_schema]}
                    if task.feature_schema
                    else {}
                ),
                **(
                    {"target_origin": task.target_origin}
                    if task.target_origin != "adapter-v1"
                    else {}
                ),
                "task_type": task.task_type,
                "classes": task.classes,
                "feature_names": [f"x{i}" for i in range(task.inputs.shape[1])],
                "context_rows": np.asarray(context.cpu().numpy(), dtype=np.int64).tolist(),
                "query_rows": np.asarray(queries.cpu().numpy(), dtype=np.int64).tolist(),
            }
        ),
    )
    truth = {"query_y": task.targets[queries], "scores": task.scores}
    if task.probabilities is not None:
        truth["probabilities"] = task.probabilities
    yield "truth.npz", _arrays(**truth)
    yield (
        "card.json",
        _json(
            {
                "source_task_id": task.source_task_id,
                "episode_id": task.episode_id,
                "recipe": json.loads(task.recipe_json),
                "mechanism": json.loads(task.mechanism_json),
            }
        ),
    )


def _digest(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def save_group(
    path: Path, tasks: Iterable[tuple[int, TrainingTask]], *, max_bytes: int
) -> GroupReceipt:
    """Never overwrite. Interrupted archives remain without a completion receipt.

    max_bytes bounds archive bytes, not receipt bytes or the caller's memory.
    The caller must persist both files to its storage provider before acknowledging
    completion remotely. Only one task's serialized array set is held at a time.
    """
    if type(max_bytes) is not int or max_bytes < 4096:
        raise ValueError("max_bytes must be an integer of at least 4096")
    if path.suffix != ".zip":
        raise ValueError("group path must end with .zip")
    marker = path.with_suffix(".json")
    if marker.exists():
        raise FileExistsError(marker)
    path.parent.mkdir(parents=True, exist_ok=True)
    ids: list[int] = []
    central_bytes = 22
    with path.open("xb") as handle:
        with zipfile.ZipFile(handle, "w", compression=zipfile.ZIP_STORED) as archive:
            for task_id, task in tasks:
                if type(task_id) is not int or not 0 <= task_id < 2**63 or task_id in ids:
                    raise ValueError("group task identities must be unique nonnegative integers")
                if json.loads(task.recipe_json)["task_id"] != task_id:
                    raise ValueError("task identity disagrees with its recipe record")
                for name, data in _entries(task):
                    name = f"{task_id}/{name}"
                    # Conservative ZIP/ZIP64 local and directory header reservation.
                    next_central = central_bytes + 128 + len(name)
                    if handle.tell() + 128 + len(name) + len(data) + next_central > max_bytes:
                        raise GroupLimitError("group archive byte limit exceeded")
                    archive.writestr(name, data)
                    central_bytes = next_central
                ids.append(task_id)
            if not ids:
                raise EmptyGroupError("cannot complete an empty group")
        handle.flush()
        os.fsync(handle.fileno())
    with zipfile.ZipFile(path) as archive:
        if archive.testzip() is not None:
            raise ValueError("archive verification failed")
    receipt = GroupReceipt(_digest(path), path.stat().st_size, tuple(ids))
    write_record(marker, asdict(receipt))
    return receipt


def verify_group(path: Path) -> GroupReceipt:
    """Refuse incomplete or modified output; do not regenerate or repair it silently."""
    marker = path.with_suffix(".json")
    if not marker.is_file():
        raise ValueError("group has no completion receipt")
    try:
        data = json.loads(marker.read_bytes())
        receipt = GroupReceipt(
            data["sha256"], data["size_bytes"], tuple(data["task_ids"]), data["version"]
        )
        if (
            receipt.version != "task-group-v1"
            or not receipt.task_ids
            or (
                any(type(v) is not int or not 0 <= v < 2**63 for v in receipt.task_ids)
                or len(set(receipt.task_ids)) != len(receipt.task_ids)
            )
        ):
            raise ValueError("invalid group receipt")
        if receipt.size_bytes != path.stat().st_size or receipt.sha256 != _digest(path):
            raise ValueError("group checksum mismatch")
        with zipfile.ZipFile(path) as archive:
            expected = {
                f"{i}/{name}"
                for i in receipt.task_ids
                for name in ("predictor.npz", "predictor.json", "truth.npz", "card.json")
            }
            if set(archive.namelist()) != expected or len(archive.namelist()) != len(expected):
                raise ValueError("group archive members do not match receipt")
            if archive.testzip() is not None:
                raise ValueError("group member checksum mismatch")
        return receipt
    except (KeyError, TypeError, json.JSONDecodeError, zipfile.BadZipFile) as error:
        raise ValueError("invalid group receipt/archive") from error
