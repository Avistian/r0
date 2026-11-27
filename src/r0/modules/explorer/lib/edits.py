"""Persist linked mechanism edits without mislabelling them causal interventions."""

import json
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from r0.modules import priors
from r0.modules.explorer.lib.saved import inside, load_saved, retain_record, run_saved
from r0.modules.generation import RunRequest


@dataclass(frozen=True)
class SavedEdit:
    before: priors.TrainingTask
    after: priors.TrainingTask
    document: dict[str, Any]
    path: str


def edit_card(before: priors.TrainingTask, after: priors.TrainingTask) -> dict[str, Any]:
    left, right = json.loads(before.recipe_json), json.loads(after.recipe_json)
    same_rows = len(before.inputs) == len(after.inputs)
    same_seed = (left["seed"], left["task_id"]) == (right["seed"], right["task_id"])
    same_target_vocabulary = (
        not (before.feature_schema or after.feature_schema) or before.classes == after.classes
    )
    return {
        "version": "mechanism-edit-card-v1",
        "kind": "mechanism-edit",
        "before_episode_id": before.episode_id,
        "after_episode_id": after.episode_id,
        "before_source_id": before.source_task_id,
        "after_source_id": after.source_task_id,
        "task_id": left["task_id"],
        "same_seed_and_task_id": same_seed,
        "same_row_count": same_rows,
        "same_split": same_rows and torch.equal(before.context_indices, after.context_indices),
        "changed_target_fraction": float(
            (before.targets != after.targets).to(torch.float64).mean().item()
        )
        if same_rows and same_target_vocabulary
        else None,
        "before_signal_std": [float(v.item()) for v in before.scores.std(dim=0, unbiased=False)],
        "after_signal_std": [float(v.item()) for v in after.scores.std(dim=0, unbiased=False)],
        "semantics": (
            "Authored mechanism edit. Common seed is retained; preservation of exogenous draws "
            "under structural changes is not asserted. This is not a causal intervention."
        ),
    }


def create_edit(
    output_dir: Path,
    before: priors.TrainingTask,
    archive: str,
    data: dict[str, Any],
    *,
    limits: priors.CausalLimits,
) -> SavedEdit:
    record = json.loads(before.recipe_json)
    recipe = priors.recipe_from_dict(data["recipe"], limits=limits)
    if not isinstance(
        recipe,
        (priors.FunctionalCausalRecipe, priors.NativeCausalRecipe, priors.NativeVectorRecipe),
    ):
        raise ValueError("mechanism edits require a frozen authored functional or native graph")
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
    after, after_archive, after_run = run_saved(output_dir, request)
    document = {
        "version": "explorer-edit-v1",
        "card": edit_card(before, after),
        "before_archive": archive,
        "after_archive": after_archive,
        "after_run": after_run,
    }
    path = output_dir / "edits" / f"{uuid.uuid4().hex}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    retain_record(path, document)
    return SavedEdit(before, after, document, str(path.relative_to(output_dir)))


def open_edit(output_dir: Path, name: str) -> SavedEdit:
    path = inside(output_dir, output_dir / name)
    if path.stat().st_size > 1024**2:
        raise ValueError("saved edit exceeds 1 MiB")
    document = json.loads(path.read_text())
    if document["version"] != "explorer-edit-v1":
        raise ValueError("unsupported saved edit version")
    task_id = document["card"]["task_id"]
    before = load_saved(output_dir, output_dir / document["before_archive"], task_id)
    after = load_saved(output_dir, output_dir / document["after_archive"], task_id)
    if edit_card(before, after) != document["card"]:
        raise ValueError("saved edit card differs from verified tasks")
    return SavedEdit(before, after, document, str(path.relative_to(output_dir)))
