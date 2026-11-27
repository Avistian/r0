"""Guarded GP target comparison construction, with complete controls before source replay."""

import json
from typing import Any

from r0.modules import priors as p


def gp_target_edit(
    before: p.TrainingTask, data: dict[str, Any], *, limits: p.CausalLimits
) -> p.TaskPair:
    record = json.loads(before.recipe_json)
    raw = record["recipe"]
    if (
        raw.get("version") != "gp-column-recipe-v1"
        or raw.get("source", {}).get("version") != "gp-column-population-v1"
    ):
        raise ValueError("select an original retained GP population before editing its target")
    target = json.loads(data["target_json"])
    mode, fresh_seed = data.get("mode", "matched"), data.get("fresh_seed")
    p.validate_gp_target_edit_controls(raw, target, mode=mode, fresh_seed=fresh_seed, limits=limits)
    if any(
        type(record.get(k)) is not int or record[k] != raw["source"][k] for k in ("seed", "task_id")
    ):
        raise ValueError("GP target edit original transport identity differs")
    recipe = p.recipe_from_dict(raw, limits=limits)
    if not isinstance(recipe, p.GPColumnRecipe):
        raise ValueError("GP target edits require an original GP recipe")
    expected = p.generate_task(recipe, seed=record["seed"], task_id=record["task_id"])
    # The public original trace validator checks semantic metadata and every
    # retained tensor, while allowing descriptive historical execution fields.
    p.explain_generation(before, [0], limits=limits)
    if expected.episode_id != before.episode_id:
        raise ValueError("GP target edit original identity differs")
    return p.generate_gp_column_pair(
        recipe,
        p.gp_column_target_from_dict(target, limits=limits),
        mode=mode,
        fresh_seed=fresh_seed,
        limits=limits,
    )
