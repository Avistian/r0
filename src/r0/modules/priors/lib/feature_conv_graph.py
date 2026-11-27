"""Convolution geometry and exact spatial ancestry without dense kernel expansion."""

from collections.abc import Callable

from r0.modules.priors.lib.causal_activations import CausalActivation
from r0.modules.priors.lib.feature_conv_records import FrozenFeatureConv1D


def convolution_geometry(function: FrozenFeatureConv1D) -> tuple[tuple[int, ...], int, int]:
    length, channels = function.input_length, function.input_channels
    widths = [length * channels]
    work = 0
    for layer in function.layers:
        output, left, right = layer.geometry.shape(length, channels)
        work = max(
            work,
            8
            * (channels * (length + left + right) + channels * layer.geometry.kernel_size * output)
            + 16 * (length * channels + output * layer.geometry.output_channels)
            + layer.activation.working_extra_channels,
        )
        length, channels = output, layer.geometry.output_channels
        widths.append(length * channels)
    return tuple(widths), work, 0


def convolution_certificates[T](
    function: FrozenFeatureConv1D,
    inputs: list[T],
    combine: Callable[[list[T]], T],
    activate: Callable[[list[T], CausalActivation], list[T]],
) -> list[T]:
    """Propagate nonzero taps, activation coupling and residual ancestry exactly."""
    length, channels = function.input_length, function.input_channels
    values = inputs

    def index(channel: int, position: int, length: int, channels: int) -> int:
        return (
            channel * length + position
            if function.layout == "channel-major"
            else position * channels + channel
        )

    def activation(
        values: list[T], length: int, channels: int, scope: str, law: CausalActivation
    ) -> list[T]:
        if scope == "flat":
            return activate(values, law)
        result = list(values)
        groups = (
            [[index(c, p, length, channels) for c in range(channels)] for p in range(length)]
            if scope == "channels"
            else [[index(c, p, length, channels) for p in range(length)] for c in range(channels)]
        )
        for group in groups:
            for i, value in zip(group, activate([values[j] for j in group], law), strict=True):
                result[i] = value
        return result

    for layer in function.layers:
        residual = values
        if layer.placement == "before":
            values = activation(values, length, channels, layer.activation_scope, layer.activation)
        output, left, _ = layer.geometry.shape(length, channels)
        outgoing = layer.geometry.output_channels
        next_values = [combine([]) for _ in range(output * outgoing)]
        for oc in range(outgoing):
            group = oc // (outgoing // layer.geometry.groups)
            for pos in range(output):
                incoming: list[T] = []
                for local, taps in enumerate(layer.kernel[oc]):
                    channel = group * (channels // layer.geometry.groups) + local
                    for tap, weight in enumerate(taps):
                        source = pos * layer.geometry.stride - left + tap * layer.geometry.dilation
                        if weight != 0 and 0 <= source < length:
                            incoming.append(values[index(channel, source, length, channels)])
                next_values[index(oc, pos, output, outgoing)] = combine(incoming)
        values = next_values
        if layer.placement == "after":
            values = activation(values, output, outgoing, layer.activation_scope, layer.activation)
        if layer.geometry.residual:
            values = [combine([a, b]) for a, b in zip(values, residual, strict=True)]
        length, channels = output, outgoing
    return values
