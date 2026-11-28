"""Original-row finite-GP target responses through the bounded document worker."""

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

from r0.modules.generation import run_guarded_document
from r0.modules.priors import CausalLimits, TrainingTask


def inspect_gp_column_curve(
    task: TrainingTask, data: dict[str, Any], output_dir: Path, *, limits: CausalLimits
) -> dict[str, Any]:
    from r0.modules.explorer.lib.curves import slice_metadata

    metadata = slice_metadata(
        task,
        data["feature"],
        data.get("anchor_row", 0),
        data.get("points", 129),
        data.get("bounds"),
        maximum=512,
    )
    record = json.loads(task.recipe_json)
    result = run_guarded_document(
        {
            "validation": "gp-column-curve",
            "recipe": record["recipe"],
            "seed": record["seed"],
            "task_id": record["task_id"],
            "feature": metadata["feature"],
            "anchor_row": metadata["anchor_row"],
            "points": metadata["points"],
            "bounds": data.get("bounds"),
            "causal_limits": asdict(limits),
            "max_seconds": data.get("max_seconds", 60),
            "max_bytes": data.get("max_bytes", 256 * 1024**2),
        },
        output_dir / "gp-target-reference",
    )
    if result.timed_out:
        raise ValueError("GP target reference timed out; saved dataset unchanged")
    if result.returncode or result.summary is None:
        detail = result.stderr.strip().splitlines()
        raise ValueError("GP target reference failed: " + (detail[-1] if detail else "failed"))
    return {
        **metadata,
        **result.summary,
        "kind": "gp_column_target_reference",
        "score_semantics": (
            "Frozen target equation on a one-column counterfactual grid; other columns fixed "
            "at the retained original row. Original independent calibration stays fixed. "
            "Regression mean excludes zero-mean Gaussian observation noise; class probabilities "
            "include task separation and offsets, with no labels drawn. This is not conditioning "
            "the joint GP feature distribution, a new population or a general DAG do-operator. "
            "Synthetic combinations, even within marginal ranges, may be unsupported by the "
            "joint feature distribution; explicit bounds may extrapolate."
        ),
    }


def reference_task(task: TrainingTask, pair: Any, data: dict[str, Any]) -> TrainingTask:
    """Choose a retained comparison side before sending its complete recipe to the guard."""
    side = data.get("gp_reference_side", "active")
    if side == "active":
        return task
    if side not in ("before", "after") or pair is None:
        raise ValueError("select an existing GP target comparison side")
    if (
        pair.document["card"].get("kind") != "gp-target-edit"
        or data.get("expected_intervention_id") != pair.after.episode_id
    ):
        raise ValueError("GP target comparison changed; reopen the selected pair")
    return pair.before if side == "before" else pair.after


def target_edit_parent(task: TrainingTask, archive: str | None, pair: Any, data: dict[str, Any]):
    side = data.get("gp_edit_parent", "active")
    if side == "active":
        return task, archive
    selected = reference_task(task, pair, {**data, "gp_reference_side": side})
    return selected, pair.document[f"{side}_archive"]
