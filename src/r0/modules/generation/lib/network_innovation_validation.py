"""Noisy architecture preparation and recompilation stay inside the guarded worker."""

import json
from dataclasses import asdict, replace
from typing import Any

from r0.modules import priors

NOTICE = (
    "Standalone network with internal row noise. The diagram is the original architecture; "
    "compiled state/noise/activation nodes are listed separately. Edit the authored source and "
    "explicit noise/importance plan. Saving freezes these equations in a linked edit; original "
    "sampled statistics and raw records remain in the original archive. "
    "This is not a causal intervention."
)


def noise_editor(data: dict[str, Any]) -> dict[str, Any]:
    maximum = data.get("max_bytes", 256 * 1024**2)
    if (
        type(maximum) is not int
        or not 1 <= maximum <= 256 * 1024**2
        or len(json.dumps(data, allow_nan=False).encode()) > maximum
    ):
        raise ValueError("network innovation editor request exceeds byte cap")
    limits = priors.CausalLimits(**data.get("causal_limits", {}))
    if data.get("validation") == "network-noise-editor":
        recipe = priors.recipe_from_dict(data["recipe"], limits=limits)
        if not isinstance(recipe, priors.NetworkInnovationRecipe) or recipe.source is None:
            raise ValueError("noise editor requires a frozen noisy network")
        source = replace(recipe.source, network=recipe.source.function)
        plan = recipe.plan
    else:
        source = priors.recipe_from_dict(data["recipe"], limits=limits)
        if not isinstance(source, priors.NetworkRecipe) or not isinstance(
            source.network, priors.FrozenNetworkFunction
        ):
            raise ValueError("noise draft requires an authored standalone network source")
        plan = priors.network_innovation_plan_from_dict(data["innovation_plan"], limits=limits)
        recipe = priors.with_network_innovations(source, plan)
    assert plan is not None
    result = {
        "kind": "mechanism-edit",
        "network": True,
        "noise": True,
        "construction": recipe.version,
        "editable_recipe": asdict(source),
        "recipe": asdict(source),
        "innovation_plan": asdict(plan),
        "compiled_recipe": asdict(recipe),
        "notice": NOTICE,
        "compiled": {
            "block_nodes": recipe.block_nodes,
            "nodes": [
                {
                    "index": i,
                    "parents": node.parents,
                    "function": node.function.version,
                    "channels": node.function.output_width,
                    "noise_scale": node.noise_scale,
                }
                for i, node in enumerate(recipe.nodes)
            ],
        },
    }
    if len(json.dumps(result, allow_nan=False).encode()) > maximum:
        raise ValueError("network innovation editor response exceeds byte cap")
    return result
