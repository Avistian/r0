"""Save and reopen a comparison without replacing the explorer's base dataset."""

import io
import json
import uuid
import zipfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from r0.modules.explorer.lib.construction import caller_limits
from r0.modules.explorer.lib.exports import replay_bundle
from r0.modules.explorer.lib.saved import inside, load_saved, retain_record, run_saved
from r0.modules.generation import RunRequest
from r0.modules.priors import (
    CausalLimits,
    TrainingTask,
    compare_pair,
    paired_recipe,
    recipe_from_dict,
)


@dataclass(frozen=True)
class SavedPair:
    before: TrainingTask
    after: TrainingTask
    document: dict[str, Any]
    path: str


def create_pair(
    output_dir: Path,
    before: TrainingTask,
    archive: str,
    run: dict[str, Any] | None,
    data: dict[str, Any],
) -> SavedPair:
    record = json.loads(before.recipe_json)
    if data.get("device", record["execution"]["device"]) != record["execution"]["device"]:
        raise ValueError("comparison must use the original dataset's generation device")
    limits = caller_limits(data)
    recipe = paired_recipe(
        recipe_from_dict(record["recipe"], limits=limits),
        intervention=data["intervention"],
        value=data["value"],
    )
    request = RunRequest(
        recipe=recipe,
        seed=record["seed"],
        start=record["task_id"],
        count=1,
        group_size=1,
        max_seconds=data.get("max_seconds", 60),
        max_bytes=data.get("max_bytes", 256 * 1024**2),
        device=record["execution"]["device"],
        causal_limits=limits,
    )
    attempts = output_dir / ".pair-attempts"
    attempts.mkdir(parents=True, exist_ok=True)
    attempt = uuid.uuid4().hex
    pending = {
        "attempt": attempt,
        "before_archive": archive,
        "before_episode_id": before.episode_id,
        "request": asdict(request),
        "intervention": data["intervention"],
        "allow_estimated_variance": data.get("allow_estimated_variance", False),
    }
    retain_record(attempts / f"{attempt}-started.json", {**pending, "status": "STARTED"})
    try:
        after, after_archive, after_run = run_saved(output_dir, request)
        card = compare_pair(
            before,
            after,
            intervention=data["intervention"],
            allow_estimated_variance=data.get("allow_estimated_variance", False),
            limits=limits,
        )
        document = {
            "version": "explorer-pair-v1",
            "card": card,
            "before_archive": archive,
            "after_archive": after_archive,
            "before_run": run,
            "after_run": after_run,
        }
        if len(json.dumps(document, sort_keys=True, allow_nan=False).encode()) > 1024**2:
            raise ValueError("saved comparison record exceeds 1 MiB")
        path = output_dir / "pairs" / f"{attempt}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        retain_record(path, document)
        retain_record(
            attempts / f"{attempt}-result.json",
            {**pending, "status": "COMPLETE", "path": str(path.relative_to(output_dir))},
        )
        return SavedPair(before, after, document, str(path.relative_to(output_dir)))
    except (OSError, RuntimeError, ValueError, TypeError, KeyError, KeyboardInterrupt) as error:
        retain_record(
            attempts / f"{attempt}-failed.json",
            {**pending, "status": "FAILED", "error": repr(error)},
        )
        raise


def open_pair(output_dir: Path, name: str, *, limits: CausalLimits | None = None) -> SavedPair:
    path = inside(output_dir, output_dir / name)
    if path.stat().st_size > 1024**2:
        raise ValueError("saved comparison record exceeds 1 MiB")
    document = json.loads(path.read_text())
    if document["version"] != "explorer-pair-v1":
        raise ValueError("unsupported saved comparison version")
    card = document["card"]
    before = load_saved(output_dir, output_dir / document["before_archive"], card["task_id"])
    after = load_saved(output_dir, output_dir / document["after_archive"], card["task_id"])
    checked = compare_pair(
        before,
        after,
        intervention=card["intervention"],
        allow_estimated_variance=card["allow_estimated_variance"],
        limits=limits,
    )
    if checked != card:
        raise ValueError("saved comparison card disagrees with verified datasets")
    return SavedPair(before, after, document, str(path.relative_to(output_dir)))


def pair_bundle(pair: SavedPair, *, max_bytes: int) -> bytes:
    """One total output limit covers the card and both ordinary replay bundles."""
    if type(max_bytes) is not int or max_bytes < 65536:
        raise ValueError("max_bytes must be at least 64 KiB")
    output = io.BytesIO()
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_STORED) as bundle:
        bundle.writestr(
            "card.json", json.dumps(pair.document["card"], sort_keys=True, allow_nan=False)
        )
        for name, task in (("before", pair.before), ("after", pair.after)):
            data = replay_bundle(task, max_bytes=max_bytes)
            if output.tell() + len(data) > max_bytes:
                raise ValueError("comparison bundle exceeds output-storage limit")
            bundle.writestr(f"{name}.zip", data)
    if output.tell() > max_bytes:
        raise ValueError("comparison bundle exceeds output-storage limit")
    return output.getvalue()
