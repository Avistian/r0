"""Only the worker loads/replays the source envelope and computes causal slices."""

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

from r0.modules import priors
from r0.modules.explorer.lib.construction import caller_limits
from r0.modules.explorer.lib.saved import inside
from r0.modules.generation import run_guarded_document


def inspect_observation_curve(
    task: priors.TrainingTask, archive: str | None, data: dict[str, Any], output_dir: Path
) -> dict[str, Any]:
    if archive is None:
        raise ValueError("save the observation task before evaluating its causal reference")
    record = json.loads(task.recipe_json)
    result = run_guarded_document(
        {
            "validation": "observation-curve",
            "archive": str(inside(output_dir, output_dir / archive)),
            "task_id": record["task_id"],
            "episode_id": task.episode_id,
            "reference": data.get("reference", "original-row"),
            "feature": data["feature"],
            "anchor_row": data.get("anchor_row", 0),
            "points": data.get("points", 129),
            "bounds": data.get("bounds"),
            "causal_limits": asdict(caller_limits(data)),
            "max_seconds": data.get("max_seconds", 60),
            "max_bytes": data.get("max_bytes", 256 * 1024**2),
        },
        output_dir / "observation-reference",
    )
    if result.timed_out:
        raise ValueError("Observation causal reference timed out; saved dataset unchanged")
    if result.returncode or result.summary is None:
        detail = result.stderr.strip().splitlines()
        raise ValueError(
            "Observation causal reference failed: "
            + (detail[-1] if detail else "validation failed")
        )
    return {"curve": result.summary}
