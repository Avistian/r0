"""Network trace replay runs in the same killable worker as root construction."""

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any, cast

from r0.modules import priors
from r0.modules.explorer.lib.construction import caller_limits
from r0.modules.explorer.lib.saved import inside
from r0.modules.generation import run_guarded_document


def inspect_graph(
    task: priors.TrainingTask, archive: str | None, output_dir: Path, data: dict[str, Any]
) -> dict[str, Any]:
    record = json.loads(task.recipe_json)
    if record["recipe"].get("version") not in (
        "standalone-network-v1",
        "network-input-view-v1",
        "network-input-view-v2",
    ):
        return priors.explain_generation(
            task,
            data["indices"],
            component=data.get("component"),
            column_start=data.get("column_start", 0),
            limits=caller_limits(data),
        )
    if archive is None:
        raise ValueError("save the network before inspecting its generation trace")
    result = run_guarded_document(
        {
            "validation": "network-graph",
            "archive": str(inside(output_dir, output_dir / archive)),
            "task_id": record["task_id"],
            "episode_id": task.episode_id,
            "indices": data["indices"],
            "component": data.get("component"),
            "column_start": data.get("column_start", 0),
            "causal_limits": asdict(caller_limits(data)),
            "max_seconds": data.get("max_seconds", 60),
            "max_bytes": data.get("max_bytes", 256 * 1024**2),
        },
        output_dir / "network-graph",
    )
    if result.timed_out:
        raise ValueError("Network trace timed out; active dataset unchanged")
    if result.returncode or result.summary is None:
        raise ValueError("Network trace failed: " + result.stderr[-2000:])
    return cast(dict[str, Any], result.summary["graph"])
