"""Archive-bound observation causal references under the existing process guard."""

import json
import math
from pathlib import Path
from typing import Any, cast

import numpy as np
import torch

from r0.modules import priors
from r0.modules.generation.lib.loading import load_group


def observation_curve(data: dict[str, Any]) -> dict[str, Any]:
    maximum, points = data.get("max_bytes", 256 * 1024**2), data.get("points", 129)
    reference = data.get("reference", "original-row")
    task_id = data.get("task_id")
    if (
        type(maximum) is not int
        or not 1 <= maximum <= 256 * 1024**2
        or type(points) is not int
        or not 2 <= points <= 512
        or reference not in ("zero", "original-row")
        or type(task_id) is not int
        or not 0 <= task_id < 2**63
    ):
        raise ValueError("observation reference requires bounded points/bytes and exact identity")
    path = Path(data["archive"])
    if path.stat().st_size > maximum:
        raise ValueError("observation reference archive exceeds byte cap")
    limits = priors.CausalLimits(**data.get("causal_limits", {}))
    for task in load_group(path):
        record = json.loads(task.recipe_json)
        if record["task_id"] != task_id:
            continue
        if (
            task.episode_id != data["episode_id"]
            or record["recipe"].get("version") != "network-observation-recipe-v1"
        ):
            raise ValueError("observation reference archive identity differs")
        row, feature = data.get("anchor_row", 0), data["feature"]
        if (
            type(row) is not int
            or not 0 <= row < len(task.inputs)
            or type(feature) is not int
            or not 0 <= feature < task.inputs.shape[1]
        ):
            raise ValueError("observation reference row/feature is outside the saved population")
        recipe = priors.recipe_from_dict(record["recipe"], limits=limits)
        if not isinstance(recipe, priors.NetworkObservationRecipe):
            raise ValueError("observation reference requires its complete envelope")
        recipe.admit(evaluation_rows=recipe.task.rows + points)
        # Full saved task validation precedes slicing, including dtype/schema/card
        # and unselected rows. Never reduce the archive to a plain runtime identity.
        priors.explain_generation(task, [row], limits=limits)
        raw_bounds = data.get("bounds")
        if raw_bounds is None:
            bounds: list[Any] = [
                float(task.inputs[:, feature].min()),
                float(task.inputs[:, feature].max()),
            ]
        elif type(raw_bounds) is list:
            bounds = cast(list[Any], raw_bounds)
        else:
            raise ValueError("observation reference bounds must be a pair")
        if (
            len(bounds) != 2
            or any(type(v) not in (int, float) or not math.isfinite(v) for v in bounds)
            or bounds[0] > bounds[1]
        ):
            raise ValueError("observation reference bounds must be finite and ordered")
        anchor = task.inputs[row].cpu()
        fraction = torch.linspace(0, 1, points, dtype=torch.float64)
        values = (1 - fraction) * bounds[0] + fraction * bounds[1]
        inputs = anchor.repeat(points, 1)
        inputs[:, feature] = values
        scores = priors.evaluate_observation_reference(
            recipe, inputs, seed=record["seed"], task_id=task_id, row=row, reference=reference
        )
        result: dict[str, Any] = {
            "version": "observation-causal-reference-v1",
            "kind": "observation_causal_reference",
            "episode_id": task.episode_id,
            "source_task_id": task.source_task_id,
            "recipe_json": task.recipe_json,
            "original_task_id": str(task_id),
            "reference": reference,
            "clamp_location": "state_and_readout",
            "observation_selection": recipe.selection_card(recipe.runtime),
            "feature": feature,
            "feature_role": "observed channel",
            "anchor_row": row,
            "original_row": row,
            "anchor_inputs": np.asarray(anchor.numpy(), dtype=np.float64).tolist(),
            "bounds": bounds,
            "range_source": "observed_min_max" if data.get("bounds") is None else "explicit",
            "points": points,
            "values": np.asarray(values.numpy(), dtype=np.float64).tolist(),
            "scores": np.asarray(scores.numpy(), dtype=np.float64).tolist(),
            "score_semantics": "raw target readouts under simultaneous state/readout clamps; "
            "hidden root row fixed; structural noise zero or original; no target adapter, "
            "conditional mean or class probabilities",
        }
        if len(json.dumps(result, allow_nan=False).encode()) > maximum:
            raise ValueError("observation reference result exceeds output byte cap")
        return result
    raise ValueError("observation reference task identity not found")
