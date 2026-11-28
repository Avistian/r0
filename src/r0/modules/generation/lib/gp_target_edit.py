"""Guarded GP target comparison construction, with complete controls before source replay."""

import json
from typing import Any

from r0.modules import priors as p


def gp_target_edit(
    before: p.TrainingTask, data: dict[str, Any], *, limits: p.CausalLimits
) -> p.TaskPair:
    record = json.loads(before.recipe_json)
    raw = record["recipe"]
    linked = raw.get("version") in ("gp-column-target-recipe-v1", "gp-column-target-link-v1")
    target = json.loads(data["target_json"])
    mode, fresh_seed = data.get("mode", "matched"), data.get("fresh_seed")
    if linked:
        if mode != "retain" or fresh_seed is not None:
            raise ValueError("GP linked edits require retain mode and no unused fresh seed")
        p.validate_gp_link_controls(raw, target, limits=limits)
        base = raw
        for _ in range(8):
            if base.get("version") != "gp-column-target-link-v1":
                break
            base = base["parent"]
        source = base["comparison"]["original"]
    else:
        if (
            raw.get("version") != "gp-column-recipe-v1"
            or raw.get("source", {}).get("version") != "gp-column-population-v1"
        ):
            raise ValueError("select an original GP population or retained target environment")
        p.validate_gp_target_edit_controls(
            raw, target, mode=mode, fresh_seed=fresh_seed, limits=limits
        )
        source = raw["source"]
    if any(type(record.get(k)) is not int or record[k] != source[k] for k in ("seed", "task_id")):
        raise ValueError("GP target edit original transport identity differs")
    recipe = p.recipe_from_dict(raw, limits=limits)
    if not isinstance(recipe, (p.GPColumnRecipe, p.GPColumnTargetRecipe, p.GPColumnTargetLink)):
        raise ValueError("GP target edits require a retained GP recipe")
    expected = p.generate_task(recipe, seed=record["seed"], task_id=record["task_id"])
    p.explain_generation(before, [0], limits=limits)
    if expected.episode_id != before.episode_id:
        raise ValueError("GP target edit original identity differs")
    replacement = p.gp_column_target_from_dict(target, limits=limits)
    if isinstance(recipe, (p.GPColumnTargetRecipe, p.GPColumnTargetLink)):
        linked_pair = p.generate_gp_link_pair(recipe, replacement, limits=limits)
        # Retain the actual parent archive, including descriptive historical execution.
        return p.TaskPair(before, linked_pair.after, linked_pair.card_json)
    return p.generate_gp_column_pair(
        recipe, replacement, mode=mode, fresh_seed=fresh_seed, limits=limits
    )
