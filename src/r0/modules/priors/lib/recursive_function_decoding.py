"""Strict complete graph envelope decoding with caller limits before sampled replay."""

from dataclasses import fields
from typing import Any, cast

from r0.modules.priors.lib.causal_limits import CausalLimits
from r0.modules.priors.lib.function_metadata import admit_function_metadata
from r0.modules.priors.lib.recursive_forest_function import FrozenRecursiveForestFunction


def recursive_function_from_dict(
    data: Any, *, limits: CausalLimits
) -> FrozenRecursiveForestFunction:
    from r0.modules.priors.lib.recursive_forest_decoding import (
        recursive_forest_from_dict,
        sampled_recursive_forest_from_dict,
    )

    admit_function_metadata(data, limits)
    try:
        if type(data) is not dict or set(cast(dict[str, Any], data)) != {
            f.name for f in fields(FrozenRecursiveForestFunction)
        }:
            raise ValueError("recursive structural envelope requires every exact field")
        data = cast(dict[str, Any], data)
        source = data["source"]
        if type(source) is not dict:
            raise ValueError("recursive structural source must be an object")
        source = cast(dict[str, Any], source)
        if source.get("version") == "sampled-recursive-forest-v1":
            # Lifecycle is rejected before restoring an independent construction stream.
            if (
                type(source.get("plan")) is not dict
                or cast(dict[str, Any], source["plan"]).get("lifecycle") != "frozen-function"
            ):
                raise ValueError("per-row reference is not a frozen structural function")
            decoded = sampled_recursive_forest_from_dict(source, limits=limits)
        else:
            decoded = recursive_forest_from_dict(source, limits=limits)
        function = FrozenRecursiveForestFunction(decoded, data["version"])
        function.admit(1, limits)
        return function
    except (KeyError, TypeError, AttributeError, OverflowError) as error:
        raise ValueError("malformed recursive structural envelope") from error
