"""Bounded shared-kernel application along feature positions, independently per row."""

import torch
from torch.nn import functional as F

from r0.modules.priors.lib.causal_limits import CausalLimits
from r0.modules.priors.lib.feature_conv_records import FeatureConv1DLayer, FrozenFeatureConv1D
from r0.modules.priors.lib.feature_conv_sampling_records import SampledFeatureConv1D
from r0.modules.priors.lib.joint_function_admission import function_record_cost

_DEFAULT_LIMITS = CausalLimits()


def admit_feature_conv1d(
    function: FrozenFeatureConv1D, rows: int, *, limits: CausalLimits = _DEFAULT_LIMITS
) -> dict[str, int]:
    if (
        type(function) is not FrozenFeatureConv1D
        or type(rows) is not int
        or rows < 1
        or type(limits) is not CausalLimits
    ):
        raise ValueError("convolution admission requires a frozen stack and positive row count")
    widths = [function.input_width]
    length, channels, workspace = function.input_length, function.input_channels, 0
    for layer in function.layers:
        output, left, right = layer.geometry.shape(length, channels)
        # Include a conservative im2col-sized backend workspace, even though the
        # implementation never materializes a dense input/output weight matrix.
        workspace = max(
            workspace,
            channels * (length + left + right) + channels * layer.geometry.kernel_size * output,
        )
        length, channels = output, layer.geometry.output_channels
        widths.append(length * channels)
    metadata, storage = function_record_cost(function, limits)
    report = limits.admit_dimensions(
        tuple(widths),
        tuple(() for _ in widths),
        rows=rows,
        calibration_rows=0,
        features=function.input_width,
        outputs=function.output_width,
        extra_parameters=metadata,
        extra_working_channels=8 * workspace,
        row_floor=1,
    )
    report["estimated_working_bytes"] += storage - 64 * metadata
    if report["estimated_working_bytes"] > limits.working_bytes:
        raise ValueError("convolution metadata and evaluation workspace exceed limit")
    return report


def _activate(layer: FeatureConv1DLayer, x: torch.Tensor, layout: str) -> torch.Tensor:
    if layer.activation_scope == "positions":
        return layer.activation.apply(x)
    if layer.activation_scope == "channels":
        return layer.activation.apply(x.transpose(1, 2)).transpose(1, 2)
    shaped = x if layout == "channel-major" else x.transpose(1, 2)
    result = layer.activation.apply(shaped.reshape(x.shape[0], -1)).reshape(shaped.shape)
    return result if layout == "channel-major" else result.transpose(1, 2)


@torch.no_grad()
def evaluate_feature_conv1d(
    function: FrozenFeatureConv1D | SampledFeatureConv1D,
    inputs: object,
    *,
    limits: CausalLimits = _DEFAULT_LIMITS,
) -> torch.Tensor:
    if isinstance(function, SampledFeatureConv1D):
        function.validate_replay(limits=limits)
        function = function.function
    if not isinstance(inputs, torch.Tensor) or inputs.ndim != 2:
        raise ValueError("convolution requires a row-by-feature tensor")
    admit_feature_conv1d(function, inputs.shape[0], limits=limits)
    if (
        inputs.shape[1] != function.input_width
        or inputs.dtype != torch.float64
        or not bool(torch.isfinite(inputs).all())
    ):
        raise ValueError("convolution requires finite float64 inputs with declared flattened width")
    x = (
        inputs.reshape(inputs.shape[0], function.input_channels, function.input_length)
        if function.layout == "channel-major"
        else inputs.reshape(
            inputs.shape[0], function.input_length, function.input_channels
        ).transpose(1, 2)
    )
    for layer in function.layers:
        residual = x
        _, left, right = layer.geometry.shape(x.shape[2], x.shape[1])
        if layer.placement == "before":
            x = _activate(layer, x, function.layout)
        if not bool(torch.isfinite(x).all()):
            raise ValueError("nonfinite convolution activation")
        x = F.conv1d(
            F.pad(x, (left, right)),
            inputs.new_tensor(layer.kernel),
            inputs.new_tensor(layer.bias),
            stride=layer.geometry.stride,
            dilation=layer.geometry.dilation,
            groups=layer.geometry.groups,
        )
        if not bool(torch.isfinite(x).all()):
            raise ValueError("nonfinite convolution affine output")
        if layer.placement == "after":
            x = _activate(layer, x, function.layout)
        if layer.geometry.residual:
            x = x + residual
        if not bool(torch.isfinite(x).all()):
            raise ValueError("nonfinite convolution output; no clipping or retry")
    if function.layout == "position-major":
        x = x.transpose(1, 2)
    return x.reshape(inputs.shape[0], -1)
