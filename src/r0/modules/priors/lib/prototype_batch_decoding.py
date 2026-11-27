"""Strict batch network request/realization decoding preserves population custody."""

from typing import Any, cast

from r0.modules.priors.lib.causal_limits import CausalLimits
from r0.modules.priors.lib.function_metadata import admit_function_metadata
from r0.modules.priors.lib.network_recipe import network_recipe_from_dict
from r0.modules.priors.lib.prototype_batch_network import (
    PrototypeBatchNetworkRecipe,
    RealizedPrototypeBatchNetworkRecipe,
)
from r0.modules.priors.lib.prototype_root_decoding import (
    prototype_batch_reference_from_dict,
    prototype_object,
    prototype_weight_law_from_dict,
    require_nested_fields,
)
from r0.modules.priors.lib.raw_draw_records import RawDrawLimits

_DEFAULT_LIMITS = CausalLimits()


def prototype_batch_network_from_dict(
    data: Any, *, limits: CausalLimits = _DEFAULT_LIMITS
) -> PrototypeBatchNetworkRecipe | RealizedPrototypeBatchNetworkRecipe:
    admit_function_metadata(data, limits)
    try:
        if type(data) is not dict:
            raise ValueError("batch network record must be an object")
        data = cast(dict[str, Any], data)
        if data.get("version") == "prototype-batch-network-realized-v1":
            fields = prototype_object(data, RealizedPrototypeBatchNetworkRecipe)
            request = prototype_batch_network_from_dict(fields["request"], limits=limits)
            if type(request) is not PrototypeBatchNetworkRecipe:
                raise ValueError("realized batch must retain its original request")
            fields["request"] = request
            fields["population"] = prototype_batch_reference_from_dict(
                fields["population"], limits=limits
            )
            return RealizedPrototypeBatchNetworkRecipe(**fields)
        fields = prototype_object(data, PrototypeBatchNetworkRecipe)
        original_source = fields["source"]
        fields["source"] = network_recipe_from_dict(original_source, limits=limits)
        require_nested_fields(original_source, fields["source"])
        fields["weights"] = prototype_weight_law_from_dict(fields["weights"])
        fields["raw_limits"] = RawDrawLimits(
            **prototype_object(fields["raw_limits"], RawDrawLimits)
        )
        return PrototypeBatchNetworkRecipe(**fields)
    except (KeyError, TypeError, AttributeError, OverflowError, RecursionError) as error:
        raise ValueError("malformed prototype batch network record") from error
