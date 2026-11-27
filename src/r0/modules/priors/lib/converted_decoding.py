"""Strict, metadata-bounded decoding for typed converter graph records."""

from dataclasses import fields
from typing import Any

from r0.modules.priors.lib.categorical_converter import CategoricalConverter
from r0.modules.priors.lib.causal_limits import CausalLimits
from r0.modules.priors.lib.converted_causal_records import (
    ConvertedCausalNode,
    ConvertedCausalRecipe,
    ConvertedReadout,
)
from r0.modules.priors.lib.frozen_function_decoding import frozen_function_from_dict
from r0.modules.priors.lib.function_metadata import admit_function_metadata
from r0.modules.priors.lib.functional_causal_decoding import (
    functional_array,
    functional_node_from_dict,
    functional_record,
)
from r0.modules.priors.lib.linear import LinearRecipe


def categorical_converter_from_dict(data: object, *, limits: CausalLimits) -> CategoricalConverter:
    args = functional_record(data, CategoricalConverter)
    for key in ("vocabulary", "mean", "scale", "bias"):
        if key in args:
            args[key] = functional_array(args[key])
    args["representatives"] = tuple(
        functional_array(row) for row in functional_array(args["representatives"])
    )
    if args.get("post_function") is not None:
        args["post_function"] = frozen_function_from_dict(args["post_function"], limits=limits)
    return CategoricalConverter(**args)


def converted_recipe_from_dict(data: object, *, limits: CausalLimits) -> ConvertedCausalRecipe:
    admit_function_metadata(data, limits)
    try:
        args: dict[str, Any] = functional_record(data, ConvertedCausalRecipe)
        declared = CausalLimits(**functional_record(args.get("limits", {}), CausalLimits))
        if any(getattr(declared, f.name) > getattr(limits, f.name) for f in fields(limits)):
            raise ValueError("converted declared limits exceed caller limits")
        admit_function_metadata(data, declared)
        task = functional_record(args["task"], LinearRecipe)
        if "class_offsets" in task:
            task["class_offsets"] = functional_array(task["class_offsets"])
        raw_nodes = functional_array(args["nodes"])
        if not 1 <= len(raw_nodes) <= declared.nodes:
            raise ValueError("converted node limit exceeded before decoding")
        nodes: list[ConvertedCausalNode] = []
        for raw in raw_nodes:
            node = functional_record(raw, ConvertedCausalNode)
            node["equation"] = functional_node_from_dict(node["equation"])
            node["converters"] = tuple(
                categorical_converter_from_dict(c, limits=declared)
                for c in functional_array(node.get("converters", ()))
            )
            nodes.append(ConvertedCausalNode(**node))
        args.update(
            task=LinearRecipe(**task),
            nodes=tuple(nodes),
            limits=declared,
            observed=tuple(
                ConvertedReadout(**functional_record(r, ConvertedReadout))
                for r in functional_array(args["observed"])
            ),
            target=ConvertedReadout(**functional_record(args["target"], ConvertedReadout)),
        )
        return ConvertedCausalRecipe(**args)
    except (KeyError, TypeError, RecursionError, OverflowError) as error:
        raise ValueError("malformed converted causal recipe") from error
