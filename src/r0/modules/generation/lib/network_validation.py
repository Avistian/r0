"""Read-only authored network admission inside the killable generation worker."""

import json
from dataclasses import asdict, replace
from typing import Any, cast

from r0.modules import priors
from r0.modules.generation.lib.network_edit_matching import apply_matching_edit_policy


def validate_network_draft(data: dict[str, Any]) -> dict[str, Any]:
    maximum = data.get("max_bytes", 256 * 1024**2)
    if type(maximum) is not int or not 1 <= maximum <= 256 * 1024**2:
        raise ValueError("network draft byte cap must be an integer in 1..268435456")
    if len(json.dumps(data, allow_nan=False).encode()) > maximum:
        raise ValueError("network draft exceeds its byte cap")
    limits = priors.CausalLimits(**data.get("causal_limits", {}))
    raw = data["recipe"]
    original = data.get("batch_source")
    if original is not None:
        verified = priors.recipe_from_dict(original, limits=limits)
        if not isinstance(verified, priors.RealizedPrototypeBatchNetworkRecipe):
            raise ValueError("batch edit source must be an original realized archive")
        if type(raw) is not dict:
            raise ValueError("batch edit draft must be an object")
        raw = cast(dict[str, Any], raw)
        source = raw.get("source")
        retained = json.loads(json.dumps(asdict(verified.request.source)))["root"]
        # Browser JSON loses integral float spelling. An unchanged sampled root
        # is restored from the strictly replayed source; edited roots must use
        # their explicit authored representation and pass the usual decoder.
        if type(source) is dict and cast(dict[str, Any], source).get("root") == retained:
            raw = {**raw, "source": {**cast(dict[str, Any], source), "root": retained}}
    recipe = priors.recipe_from_dict(raw, limits=limits)
    if isinstance(recipe, priors.RealizedPrototypeBatchNetworkRecipe):
        recipe = recipe.request
    if isinstance(recipe, priors.PrototypeBatchNetworkRecipe):
        if data.get("validation") == "network-editor":
            recipe = replace(recipe, source=replace(recipe.source, network=recipe.source.function))
        elif not isinstance(recipe.source.network, priors.FrozenNetworkFunction):
            raise ValueError("batch network edits require an authored frozen function")
        return {"status": "VALIDATED", "recipe": asdict(recipe)}
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
