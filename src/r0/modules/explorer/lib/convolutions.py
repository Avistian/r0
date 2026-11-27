"""Convolution inspection remains bounded; raw receipt detection never replays RNG."""

import json
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any, cast

from r0.modules import priors as p
from r0.modules.explorer.lib.construction import caller_limits
from r0.modules.explorer.lib.interventions import intervention_controls
from r0.modules.generation import run_guarded_document


def contains_convolution(value: Any) -> bool:
    if type(value) is dict:
        raw = cast(dict[str, Any], value)
        return raw.get("version") in (
            "sampled-feature-conv1d-r0-v1",
            "frozen-feature-conv1d-r0-v1",
        ) or any(contains_convolution(v) for v in raw.values())
    if type(value) in (list, tuple):
        return any(contains_convolution(v) for v in cast(list[Any] | tuple[Any, ...], value))
    return False


def convolution_action(
    action: str, task: p.TrainingTask, data: dict[str, Any], output_dir: Path
) -> dict[str, Any] | None:
    if action not in ("mechanism", "convolution_draft", "intervention_controls", "curve"):
        return None
    original = json.loads(task.recipe_json)["recipe"]
    if not contains_convolution(original):
        return None
    if action == "curve":
        if original["version"] in ("functional-causal-v1", "functional-causal-v2"):
            return None
        raise ValueError(
            "Typed convolution reference curves require explicit category/capture semantics "
            "and remain pending; use original-row traces and interventions"
        )
    result = run_guarded_document(
        {
            "validation": "convolution-draft"
            if action == "convolution_draft"
            else "convolution-editor",
            "recipe": data["recipe"] if action == "convolution_draft" else original,
            "causal_limits": asdict(caller_limits(data)),
            "max_seconds": data.get("max_seconds", 60),
            "max_bytes": data.get("max_bytes", 256 * 1024**2),
        },
        output_dir / "convolution-editor",
    )
    if result.timed_out:
        raise ValueError("Convolution inspection timed out; active dataset unchanged")
    if result.returncode or result.summary is None:
        raise ValueError("Convolution inspection failed: " + result.stderr[-2000:])
    if action != "intervention_controls":
        return result.summary
    editable = replace(task, recipe_json=json.dumps({"recipe": result.summary["editable_recipe"]}))
    info = intervention_controls(editable)
    info["notice"] = result.summary["notice"]
    return info
