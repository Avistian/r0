"""Read-only authored network admission inside the killable generation worker."""

import json
from dataclasses import asdict, replace
from typing import Any

from r0.modules import priors
from r0.modules.generation.lib.network_edit_matching import apply_matching_edit_policy


def validate_network_draft(data: dict[str, Any]) -> dict[str, Any]:
    maximum = data.get("max_bytes", 256 * 1024**2)
    if type(maximum) is not int or not 1 <= maximum <= 256 * 1024**2:
        raise ValueError("network draft byte cap must be an integer in 1..268435456")
    if len(json.dumps(data, allow_nan=False).encode()) > maximum:
        raise ValueError("network draft exceeds its byte cap")
    limits = priors.CausalLimits(**data.get("causal_limits", {}))
    recipe = priors.recipe_from_dict(data["recipe"], limits=limits)
    base = recipe.base if isinstance(recipe, priors.NetworkInputView) else recipe
    if data.get("validation") == "network-editor":
        if not isinstance(base, priors.NetworkRecipe):
            raise ValueError("network editor requires a network recipe")
        authored = replace(base, network=base.function)
        editable = (
            priors.NetworkInputView(authored, recipe.task, recipe.active_inputs)
            if isinstance(recipe, priors.NetworkInputView)
            else authored
        )
        return {"status": "VALIDATED", "recipe": asdict(editable)}
    if not isinstance(base, priors.NetworkRecipe) or not isinstance(
        base.network, priors.FrozenNetworkFunction
    ):
        raise ValueError("network drafts require an authored frozen network recipe")
    recipe = apply_matching_edit_policy(recipe, data, limits)
    return {"status": "VALIDATED", "recipe": asdict(recipe)}
