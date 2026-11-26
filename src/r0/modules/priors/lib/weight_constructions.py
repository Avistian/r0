"""Bounded decoder and shared type registry for frozen matrix construction stages."""

from typing import Any

from r0.modules.priors.lib.causal_activations import CausalActivation
from r0.modules.priors.lib.transformed_weights import (
    SIMPLE_WEIGHT_TYPES,
    GaussianWeights,
    NormalizedWeights,
    SimpleWeights,
    TransformedWeights,
)
from r0.modules.priors.lib.weight_matrices import (
    FactorizedWeights,
    ImportanceWeights,
    LaplaceKernelWeights,
)
from r0.modules.priors.lib.weight_vectors import weight_vector_from_dict

type WeightConstruction = SimpleWeights | NormalizedWeights | TransformedWeights
WEIGHT_CONSTRUCTION_TYPES = (*SIMPLE_WEIGHT_TYPES, NormalizedWeights, TransformedWeights)


def weight_construction_from_dict(data: dict[str, Any], *, _depth: int = 0) -> WeightConstruction:
    if _depth > 3:
        raise ValueError("matrix construction nesting exceeds the nonrecursive stage limit")
    fields = dict(data)
    version = fields.get("version")
    if version == "gaussian-weights-v1":
        if "draws" in fields:
            fields["draws"] = tuple(tuple(row) for row in fields["draws"])
        return GaussianWeights(**fields)
    if version in ("normalized-weights-v1", "transformed-weights-v1"):
        if "jitter" in fields:
            fields["jitter"] = tuple(tuple(row) for row in fields["jitter"])
        if version == "normalized-weights-v1":
            if "source" in fields:
                fields["source"] = weight_construction_from_dict(
                    fields["source"], _depth=_depth + 1
                )
            return NormalizedWeights(**fields)
        if "child" in fields:
            fields["child"] = weight_construction_from_dict(fields["child"], _depth=_depth + 1)
        if "activation" in fields:
            activation = dict(fields["activation"])
            activation["parameters"] = tuple(
                tuple(pair) for pair in activation.get("parameters", ())
            )
            fields["activation"] = CausalActivation(**activation)
        return TransformedWeights(**fields)
    if fields.get("version") == "laplace-kernel-weights-v1":
        for name in ("left_points", "right_points", "signs"):
            if name in fields:
                fields[name] = tuple(tuple(row) for row in fields[name])
        return LaplaceKernelWeights(**fields)
    if fields.get("version") == "importance-weights-v1":
        fields["gaussian"] = tuple(tuple(row) for row in fields["gaussian"])
        fields["importance"] = tuple(weight_vector_from_dict(v) for v in fields["importance"])
        return ImportanceWeights(**fields)
    for name in ("left", "right"):
        if name in fields:
            fields[name] = tuple(tuple(row) for row in fields[name])
    if "diagonal" in fields:
        fields["diagonal"] = tuple(fields["diagonal"])
    return FactorizedWeights(**fields)
