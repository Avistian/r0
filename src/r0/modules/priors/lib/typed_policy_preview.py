"""Validation-only authored graph views never replace retained policy bases."""

from dataclasses import fields
from typing import Any, cast

from r0.modules.priors.lib.causal_limits import CausalLimits
from r0.modules.priors.lib.converted_causal_records import ConvertedCausalRecipe
from r0.modules.priors.lib.functional_causal import FunctionalCausalRecipe
from r0.modules.priors.lib.native_causal import NativeCausalRecipe
from r0.modules.priors.lib.native_vector_records import NativeVectorRecipe
from r0.modules.priors.lib.recursive_task_preflight import recursive_authored_preview


def _roots(value: Any) -> Any:
    if type(value) is dict:
        raw = cast(dict[str, Any], value)
        if raw.get("version") == "sampled-source-root-base-v1":
            return {
                "law": raw["law"],
                "calibration": raw["calibration"],
                "version": "source-root-base-v1",
            }
        if raw.get("version") == "sampled-prototype-root-v1":
            return _roots(raw["root"])
        return {key: _roots(item) for key, item in raw.items()}
    if type(value) in (list, tuple):
        return tuple(_roots(item) for item in cast(list[Any] | tuple[Any, ...], value))
    return value


def typed_policy_base_preview(raw: dict[str, Any], limits: CausalLimits):
    """Use retained authored equations/domains, with no construction or fit replay."""
    from r0.modules.priors.lib.generate import recipe_from_dict

    graph = raw
    while "nodes" not in graph:
        nested = graph.get("runtime", graph.get("recipe", graph.get("graph")))
        if type(nested) is not dict:
            raise ValueError("typed policy requires original graph metadata")
        graph = cast(dict[str, Any], nested)
    functions = tuple(node.get("equation", node)["function"] for node in graph["nodes"])
    preview = typed_authored_preview(graph)
    if graph["version"] in (
        "fitted-functional-causal-v1",
        "sampled-fitted-functional-causal-v1",
        "xgboost-fitted-functional-causal-v1",
    ):
        preview = {field.name: preview[field.name] for field in fields(FunctionalCausalRecipe)}
        preview["version"] = "functional-causal-v1"
    result = recipe_from_dict(preview, limits=limits)
    if not isinstance(
        result,
        (FunctionalCausalRecipe, NativeCausalRecipe, NativeVectorRecipe, ConvertedCausalRecipe),
    ):
        raise ValueError("typed policy requires a supported retained graph")
    return result, functions


def typed_authored_preview(value: Any) -> Any:
    """Discard construction wrappers only for metadata validation, never for replay."""
    return _roots(recursive_authored_preview(value))
