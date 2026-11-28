"""Strict convolution custody is checked before offering an authored graph draft."""

import json
from dataclasses import asdict
from typing import Any, cast

from r0.modules import priors as p
from r0.modules.generation.lib.authored_xgboost import prepare_authored_xgboost


def _authored(value: Any) -> Any:
    if type(value) is dict:
        raw = cast(dict[str, Any], value)
        if raw.get("version") == "sampled-feature-conv1d-r0-v1":
            return raw["function"]
        return {k: _authored(v) for k, v in raw.items()}
    if type(value) in (list, tuple):
        return [_authored(v) for v in cast(list[Any] | tuple[Any, ...], value)]
    return value


def convolution_editor(data: dict[str, Any]) -> dict[str, Any]:
    maximum = data.get("max_bytes", 256 * 1024**2)
    if type(maximum) is not int or not 1 <= maximum <= 256 * 1024**2:
        raise ValueError("convolution editor requires a bounded byte cap")
    if len(json.dumps(data, allow_nan=False).encode()) > maximum:
        raise ValueError("convolution editor request exceeds byte cap")
    limits = p.CausalLimits(**data.get("causal_limits", {}))
    if data["validation"] == "convolution-draft":
        prepare_authored_xgboost(data, "authored-mechanism-v1")
    recipe = p.recipe_from_dict(data["recipe"], limits=limits)
    if type(recipe) not in (
        p.FunctionalCausalRecipe,
        p.NativeCausalRecipe,
        p.NativeVectorRecipe,
        p.ConvertedCausalRecipe,
    ):
        raise ValueError("convolution editor requires an authored or sampled-kernel graph")
    original = asdict(recipe)
    editable = _authored(original)
    if data["validation"] == "convolution-draft" and json.dumps(
        original, sort_keys=True
    ) != json.dumps(editable, sort_keys=True):
        raise ValueError(
            "edited convolution kernels must be authored, without forged sampling custody"
        )
    verified = p.recipe_from_dict(editable, limits=limits)
    raw = asdict(verified)
    native = isinstance(verified, (p.NativeCausalRecipe, p.NativeVectorRecipe))
    converted = isinstance(verified, p.ConvertedCausalRecipe)
    nodes: list[dict[str, Any]] = []
    for i, record in enumerate(raw["nodes"]):
        equation = record.get("equation", record)
        function = p.frozen_function_from_dict(equation["function"], limits=limits)
        nodes.append(
            {
                "index": i,
                "parents": equation["parents"],
                "width": function.output_width,
                "function": function.version,
                "source_function": original["nodes"][i].get("equation", original["nodes"][i])[
                    "function"
                ],
                "converters": record.get("converters", ()),
                "domain": record.get("domains", record.get("domain")),
            }
        )
    result = {
        "kind": "mechanism-edit",
        "convolution": True,
        "native": native,
        "vector": isinstance(verified, p.NativeVectorRecipe),
        "converted": converted,
        "construction": recipe.version,
        "editable_recipe": raw,
        "geometry": None,
        "nodes": nodes,
        "notice": "Shared kernels convolve feature positions within each row. Original initializer "
        "draws remain in the before archive. Kernel edits create authored functions, "
        "not resampled construction. O'Prior motivates the feature operation; geometry "
        "and initializer catalogs here are explicit r0 choices.",
    }
    if len(json.dumps(result, allow_nan=False).encode()) > maximum:
        raise ValueError("convolution editor result exceeds byte cap")
    return result
