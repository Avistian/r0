"""Admitted decoding of native typed graph recipes."""

from dataclasses import fields as dataclass_fields
from typing import Any, cast

from r0.modules.priors.lib.causal_limits import CausalLimits
from r0.modules.priors.lib.function_metadata import admit_function_metadata
from r0.modules.priors.lib.functional_causal_decoding import (
    functional_array,
    functional_node_from_dict,
    functional_record,
)
from r0.modules.priors.lib.linear import LinearRecipe
from r0.modules.priors.lib.native_causal import (
    FrequencyVariableDomain,
    NativeCausalNode,
    NativeCausalRecipe,
    VariableDomain,
)


def native_recipe_from_dict(data: dict[str, Any], *, limits: CausalLimits) -> NativeCausalRecipe:
    admit_function_metadata(data, limits)
    try:
        fields = functional_record(data, NativeCausalRecipe)
        declared = CausalLimits(**functional_record(fields.get("limits", {}), CausalLimits))
        if any(
            getattr(declared, f.name) > getattr(limits, f.name) for f in dataclass_fields(limits)
        ):
            raise ValueError("native archive exceeds caller limits; supply explicit trusted limits")
        admit_function_metadata(data, declared)
        task = functional_record(fields["task"], LinearRecipe)
        if "class_offsets" in task:
            task["class_offsets"] = functional_array(task["class_offsets"])
        fields["task"] = LinearRecipe(**task)
        raw_nodes = functional_array(fields["nodes"])
        if len(raw_nodes) > declared.nodes:
            raise ValueError("native node limit exceeded")
        nodes: list[NativeCausalNode] = []
        for raw in raw_nodes:
            node = functional_record(raw, NativeCausalNode)
            node["equation"] = functional_node_from_dict(node["equation"], limits=declared)
            raw_domain = node.get("domain", {})
            if type(raw_domain) is not dict:
                raise ValueError("native variable domain must be an object")
            raw_domain = cast(dict[str, Any], raw_domain)
            kind = (
                FrequencyVariableDomain
                if raw_domain.get("version") == "native-frequency-domain-v2"
                else VariableDomain
            )
            domain = functional_record(raw_domain, kind)
            for key in ("vocabulary", "thresholds", "frequencies"):
                if key in domain:
                    domain[key] = functional_array(domain[key])
            node["domain"] = kind(**domain)
            nodes.append(NativeCausalNode(**node))
        fields["nodes"] = tuple(nodes)
        fields["observed"] = tuple(
            functional_array(ref) for ref in functional_array(fields["observed"])
        )
        fields["target"] = functional_array(fields["target"])
        fields["limits"] = declared
        return NativeCausalRecipe(**fields)
    except (KeyError, TypeError) as error:
        raise ValueError("malformed native graph recipe") from error
