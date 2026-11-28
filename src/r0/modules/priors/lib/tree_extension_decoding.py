"""Bounded JSON receipt reopening for sampled tree extensions."""

from dataclasses import fields
from typing import Any

from r0.modules.priors.lib.causal_limits import CausalLimits
from r0.modules.priors.lib.fitted_tree_decoding import fitted_object
from r0.modules.priors.lib.functional_causal import FunctionalCausalRecipe
from r0.modules.priors.lib.functional_causal_decoding import functional_root
from r0.modules.priors.lib.generator_snapshot import GeneratorSnapshot
from r0.modules.priors.lib.linear import LinearRecipe
from r0.modules.priors.lib.network_recipe import NetworkRecipe
from r0.modules.priors.lib.observation_envelope_admission import observation_metadata_cost
from r0.modules.priors.lib.tree_extension_sampling import SampledTreeExtension, TreeExtensionRequest

_DEFAULT = CausalLimits()


def sampled_tree_extension_from_dict(
    data: Any,
    *,
    limits: CausalLimits = _DEFAULT,
) -> SampledTreeExtension:
    from r0.modules.priors.lib.generate import recipe_from_dict

    observation_metadata_cost(data, limits)
    raw = fitted_object(data, SampledTreeExtension)
    declared = CausalLimits(**fitted_object(raw["limits"], CausalLimits))
    if any(getattr(declared, f.name) > getattr(limits, f.name) for f in fields(limits)):
        raise ValueError("tree receipt declared limits exceed caller limits")
    request = tree_extension_request_from_dict(raw["request"], limits=declared)
    recipe = recipe_from_dict(raw["recipe"], limits=declared)
    if not isinstance(recipe, (FunctionalCausalRecipe, NetworkRecipe)):
        raise ValueError("tree extension receipt requires a compatible compiled recipe")
    result = SampledTreeExtension(
        request,
        recipe,
        raw["family"],
        raw["depth"],
        GeneratorSnapshot(**fitted_object(raw["initial"], GeneratorSnapshot)),
        GeneratorSnapshot(**fitted_object(raw["final"], GeneratorSnapshot)),
        declared,
        raw["version"],
    )
    result.validate_replay()
    return result


def tree_extension_request_from_dict(
    data: Any, *, limits: CausalLimits = _DEFAULT
) -> TreeExtensionRequest:
    observation_metadata_cost(data, limits)
    request_data = fitted_object(data, TreeExtensionRequest)
    task = fitted_object(request_data["task"], LinearRecipe)
    if type(task["class_offsets"]) not in (tuple, list):
        raise ValueError("tree receipt class offsets must be an array")
    task["class_offsets"] = tuple(task["class_offsets"])
    request_data["task"] = LinearRecipe(**task)
    for key in ("families", "leaf_families", "temperature_bounds"):
        if type(request_data[key]) not in (tuple, list):
            raise ValueError("tree receipt request choices must be arrays")
        request_data[key] = tuple(request_data[key])
    if request_data["root"] is not None:
        request_data["root"] = functional_root(request_data["root"], limits=limits)
    request = TreeExtensionRequest(**request_data)
    request.admit(limits)
    return request
