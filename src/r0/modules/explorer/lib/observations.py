"""Guarded receipt-derived observation graph editor and intervention controls."""

import json
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

from r0.modules import priors
from r0.modules.explorer.lib.construction import caller_limits
from r0.modules.explorer.lib.interventions import intervention_controls
from r0.modules.generation import run_guarded_document


def inspect_observation(
    task: priors.TrainingTask,
    data: dict[str, Any],
    output_dir: Path,
    *,
    controls: bool = False,
    draft: bool = False,
) -> dict[str, Any]:
    raw = json.loads(task.recipe_json)["recipe"]
    result = run_guarded_document(
        {
            "validation": "observation-draft" if draft else "observation-editor",
            "recipe": data["recipe"] if draft else raw,
            "causal_limits": asdict(caller_limits(data)),
            "max_seconds": data.get("max_seconds", 60),
            "max_bytes": data.get("max_bytes", 256 * 1024**2),
        },
        output_dir / "observation-editor",
    )
    if result.timed_out:
        raise ValueError("Observation inspection timed out; active dataset unchanged")
    if result.returncode or result.summary is None:
        raise ValueError("Observation inspection failed: " + result.stderr[-2000:])
    if not controls:
        return result.summary
    # This helper reads serialized node fields only; no source decoder runs in the parent.
    runtime_task = replace(
        task, recipe_json=json.dumps({"recipe": result.summary["editable_recipe"]})
    )
    info = intervention_controls(runtime_task)
    info["observation_selection"] = result.summary["observation_selection"]
    info["notice"] = result.summary["notice"]
    return info


def observation_action(
    action: str, task: priors.TrainingTask, data: dict[str, Any], output_dir: Path
) -> dict[str, Any] | None:
    if action not in ("mechanism", "intervention_controls", "observation_draft", "curve"):
        return None
    if json.loads(task.recipe_json)["recipe"].get("version") != "network-observation-recipe-v1":
        return None
    if action == "curve":
        raise ValueError(
            "Observation causal-reference curves remain required follow-on work; "
            "use original-row traces or explicit interventions"
        )
    return inspect_observation(
        task,
        data,
        output_dir,
        controls=action == "intervention_controls",
        draft=action == "observation_draft",
    )
