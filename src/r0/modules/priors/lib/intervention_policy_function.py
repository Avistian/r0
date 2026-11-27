"""Authored replacements leave sampled construction receipts in the original base."""

from typing import Any, cast

from r0.modules.priors.lib.causal_limits import CausalLimits
from r0.modules.priors.lib.feature_conv_records import FrozenFeatureConv1D
from r0.modules.priors.lib.feature_conv_sampling_records import SampledFeatureConv1D
from r0.modules.priors.lib.frozen_function_decoding import frozen_function_from_dict
from r0.modules.priors.lib.frozen_function_evaluation import FrozenFunction


def finish_policy_function(
    original: FrozenFunction,
    record: dict[str, Any],
    paths: tuple[tuple[str, ...], ...],
    limits: CausalLimits,
) -> FrozenFunction:
    """Unwrap only sampled convolutions traversed by the declared parameter paths.

    Paths still address the complete original envelope. Its raw draws and RNG
    snapshots stay in request.base; the changed kernel is an authored function,
    never a claim that different weights came from the original initializer.
    Unselected sampled siblings retain their complete replay-bound envelopes.
    """

    def authored(value: Any, selected: tuple[tuple[str, ...], ...]) -> Any:
        if not selected:
            return value
        if type(value) is dict:
            fields = cast(dict[str, Any], value)
            if fields.get("version") == "sampled-feature-conv1d-r0-v1":
                if any(not path or path[0] != "function" for path in selected):
                    raise ValueError("policy laws cannot alter sampled convolution custody")
                return authored(fields["function"], tuple(path[1:] for path in selected))
            return {
                key: authored(item, tuple(path[1:] for path in selected if path and path[0] == key))
                for key, item in fields.items()
            }
        if type(value) is list:
            return [
                authored(item, tuple(path[1:] for path in selected if path and path[0] == str(i)))
                for i, item in enumerate(cast(list[Any], value))
            ]
        return value

    function = frozen_function_from_dict(authored(record, paths), limits=limits)
    same_family = type(function) is type(original) or (
        type(original) is SampledFeatureConv1D and type(function) is FrozenFeatureConv1D
    )
    if (
        not same_family
        or function.input_width != original.input_width
        or function.output_width != original.output_width
    ):
        raise ValueError("soft resampling must preserve function family and geometry")
    return function
