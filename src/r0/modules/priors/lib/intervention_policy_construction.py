"""Original initializer provenance alongside explicitly authored policy replacements."""

from collections.abc import Callable
from typing import Any

from r0.modules.priors.lib.feature_conv_construction_steps import convolution_construction_steps
from r0.modules.priors.lib.feature_conv_sampling_records import SampledFeatureConv1D
from r0.modules.priors.lib.frozen_function_evaluation import FrozenFunction
from r0.modules.priors.lib.intervention_policy_laws import parameter_group


def policy_construction_steps(
    original: FrozenFunction,
    paths: tuple[tuple[str, ...], ...],
    *,
    node: int,
    add: Callable[..., None],
) -> list[dict[str, Any]]:
    """Expose historical draws without claiming they generated overwritten weights."""
    found: dict[tuple[str, ...], SampledFeatureConv1D] = {}
    for path in paths:
        for length in range(len(path)):
            prefix = path[:length]
            value = original if not prefix else parameter_group(original, prefix)[2]
            if isinstance(value, SampledFeatureConv1D):
                found[prefix] = value
    result: list[dict[str, Any]] = []
    for index, (prefix, sampled) in enumerate(found.items()):
        identifiers = convolution_construction_steps(
            sampled, prefix=f"policy-original-{node}-{index}", add=add
        )
        result.append(
            {
                "original_path": prefix,
                "original_version": sampled.version,
                "replacement_version": sampled.function.version,
                "construction_parameter_nodes": list(identifiers.values()),
                "replaced_paths": [path for path in paths if path[: len(prefix)] == prefix],
                "semantics": "historical initializer receipt in the unchanged base; replacement "
                "parameters use explicit r0 policy draws, not a replay of this initializer",
            }
        )
    return result
