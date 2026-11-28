"""Immutable feature-axis kernels with explicit geometry and flattened layouts."""

from dataclasses import dataclass

from r0.modules.priors.lib.activation_sampling import finite_number
from r0.modules.priors.lib.causal_activations import CausalActivation


@dataclass(frozen=True, slots=True)
class Conv1DGeometry:
    output_channels: int
    kernel_size: int
    stride: int = 1
    dilation: int = 1
    padding: str = "valid"
    groups: int = 1
    explicit_padding: tuple[int, int] = (0, 0)
    residual: bool = False
    version: str = "feature-conv1d-geometry-r0-v1"

    def __post_init__(self) -> None:
        if (
            self.version != "feature-conv1d-geometry-r0-v1"
            or any(
                type(v) is not int or not 1 <= v < 2**31
                for v in (
                    self.output_channels,
                    self.kernel_size,
                    self.stride,
                    self.dilation,
                    self.groups,
                )
            )
            or self.output_channels % self.groups
            or self.padding not in ("valid", "same-upper", "explicit")
            or type(self.explicit_padding) is not tuple
            or len(self.explicit_padding) != 2
            or any(type(v) is not int or not 0 <= v < 2**31 for v in self.explicit_padding)
            or self.padding != "explicit"
            and self.explicit_padding != (0, 0)
            or type(self.residual) is not bool
        ):
            raise ValueError("invalid explicit feature convolution geometry")

    def shape(self, length: int, channels: int) -> tuple[int, int, int]:
        if any(type(v) is not int or v < 1 for v in (length, channels)) or channels % self.groups:
            raise ValueError("convolution input channels must divide into declared groups")
        effective = self.dilation * (self.kernel_size - 1) + 1
        left, right = self.explicit_padding if self.padding == "explicit" else (0, 0)
        if self.padding == "same-upper":
            total = max(
                0,
                ((length + self.stride - 1) // self.stride - 1) * self.stride + effective - length,
            )
            left, right = total // 2, total - total // 2
        output = (length + left + right - effective) // self.stride + 1
        if output < 1 or self.residual and (output != length or self.output_channels != channels):
            raise ValueError(
                "convolution has no output positions or incompatible residual geometry"
            )
        return output, left, right


@dataclass(frozen=True, slots=True)
class FeatureConv1DLayer:
    geometry: Conv1DGeometry
    kernel: tuple[tuple[tuple[float, ...], ...], ...]
    bias: tuple[float, ...]
    activation: CausalActivation = CausalActivation()
    placement: str = "after"
    activation_scope: str = "flat"
    version: str = "feature-conv1d-layer-r0-v1"

    def __post_init__(self) -> None:
        if (
            self.version != "feature-conv1d-layer-r0-v1"
            or type(self.geometry) is not Conv1DGeometry
            or type(self.kernel) is not tuple
            or len(self.kernel) != self.geometry.output_channels
            or not self.kernel
            or type(self.kernel[0]) is not tuple
            or not self.kernel[0]
            or any(
                type(output) is not tuple
                or len(output) != len(self.kernel[0])
                or any(
                    type(taps) is not tuple
                    or len(taps) != self.geometry.kernel_size
                    or any(not finite_number(v) for v in taps)
                    for taps in output
                )
                for output in self.kernel
            )
            or type(self.bias) is not tuple
            or len(self.bias) != len(self.kernel)
            or any(not finite_number(v) for v in self.bias)
            or type(self.activation) is not CausalActivation
            or self.placement not in ("before", "after")
            or self.activation_scope not in ("flat", "channels", "positions")
        ):
            raise ValueError("convolution requires finite aligned shared kernels and activation")

    @property
    def input_channels(self) -> int:
        return len(self.kernel[0]) * self.geometry.groups

    @property
    def parameter_count(self) -> int:
        return (
            len(self.kernel) * len(self.kernel[0]) * self.geometry.kernel_size
            + len(self.bias)
            + self.activation.parameter_count
        )


@dataclass(frozen=True, slots=True)
class FrozenFeatureConv1D:
    input_length: int
    input_channels: int
    layers: tuple[FeatureConv1DLayer, ...]
    layout: str = "channel-major"
    version: str = "frozen-feature-conv1d-r0-v1"

    def __post_init__(self) -> None:
        if (
            self.version != "frozen-feature-conv1d-r0-v1"
            or any(
                type(v) is not int or not 1 <= v < 2**31
                for v in (self.input_length, self.input_channels)
            )
            or type(self.layers) is not tuple
            or not self.layers
            or any(type(layer) is not FeatureConv1DLayer for layer in self.layers)
            or self.layout not in ("channel-major", "position-major")
        ):
            raise ValueError("invalid frozen feature convolution stack")
        length, channels = self.input_length, self.input_channels
        for layer in self.layers:
            if layer.input_channels != channels:
                raise ValueError("convolution stack channel geometry differs")
            length, _, _ = layer.geometry.shape(length, channels)
            channels = layer.geometry.output_channels

    @property
    def parameter_count(self) -> int:
        return sum(layer.parameter_count for layer in self.layers)

    @property
    def input_width(self) -> int:
        return self.input_length * self.input_channels

    @property
    def output_width(self) -> int:
        length, channels = self.input_length, self.input_channels
        for layer in self.layers:
            length, _, _ = layer.geometry.shape(length, channels)
            channels = layer.geometry.output_channels
        return length * channels
