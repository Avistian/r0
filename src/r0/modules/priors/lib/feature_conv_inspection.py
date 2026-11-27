"""Shared convolution arithmetic emitted as original-row affine and activation stages."""

from collections.abc import Callable, Sequence
from dataclasses import asdict

import torch

from r0.modules.priors.lib.feature_conv_evaluation import evaluate_admitted_feature_conv1d
from r0.modules.priors.lib.feature_conv_records import FrozenFeatureConv1D
from r0.modules.priors.lib.feature_conv_sampling_records import SampledFeatureConv1D


def feature_conv_steps(
    function: FrozenFeatureConv1D | SampledFeatureConv1D,
    inputs: torch.Tensor,
    *,
    prefix: str,
    parents: list[str],
    indices: Sequence[int],
    add: Callable[..., None],
) -> list[str]:
    from r0.modules.priors.lib.feature_conv_construction_steps import convolution_construction_steps

    parameter_ids: dict[int, str] = {}
    if isinstance(function, SampledFeatureConv1D):
        function.validate_replay()
        parameter_ids = convolution_construction_steps(function, prefix=prefix, add=add)
        function = function.function
    current = list(parents)
    incoming = list(parents)
    last_layer = -1

    def observe(index: int, stage: str, values: torch.Tensor) -> None:
        nonlocal current, incoming, last_layer
        if index != last_layer:
            incoming, last_layer = list(current), index
        identifier = f"{prefix}-layer-{index}-{stage}"
        dependencies = (
            list(dict.fromkeys([*current, *incoming])) if stage == "residual" else current
        )
        if index in parameter_ids and stage != "residual":
            dependencies = [*dependencies, parameter_ids[index]]
        selected = values[list(indices)]
        if function.layout == "position-major":
            selected = selected.transpose(1, 2)
        flattened = selected.reshape(len(indices), -1)
        layer = function.layers[index]
        add(
            identifier,
            f"Feature convolution layer {index}: {stage}",
            dependencies,
            "shared cross-correlation kernel * padded feature positions + bias"
            if stage == "affine"
            else "activated convolution + unchanged incoming layer state"
            if stage == "residual"
            else "activation over declared within-row scope",
            flattened,
            {
                "original_rows": list(indices),
                "layout": function.layout,
                "channels": values.shape[1],
                "positions": values.shape[2],
                "stage": stage,
                "layer": asdict(layer),
            },
        )
        current = [identifier]

    evaluate_admitted_feature_conv1d(function, inputs, observe=observe)
    return current
