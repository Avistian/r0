"""Explicit r0 convolution catalogs and replay-bound kernel initialization laws."""

from dataclasses import dataclass

from r0.modules.priors.lib.activation_sampling import finite_number
from r0.modules.priors.lib.causal_activations import CausalActivation
from r0.modules.priors.lib.causal_limits import CausalLimits
from r0.modules.priors.lib.feature_conv_records import Conv1DGeometry, FrozenFeatureConv1D
from r0.modules.priors.lib.generator_snapshot import GeneratorSnapshot
from r0.modules.priors.lib.raw_draw_records import RawDrawLimits, RawDrawRecord

LAWS = ("normal", "uniform", "student_t", "cauchy", "rademacher")


@dataclass(frozen=True, slots=True)
class Conv1DLayerPrior:
    geometries: tuple[Conv1DGeometry, ...]
    activations: tuple[CausalActivation, ...] = (CausalActivation("tanh"), CausalActivation())
    kernel_laws: tuple[str, ...] = ("normal",)
    kernel_scale: float = 1.0
    kernel_mean: float = 0.0
    fan_in: bool = True
    bias_law: str = "normal"
    bias_scale: float = 0.2
    bias_mean: float = 0.0
    student_df: float = 3.0
    placement: str = "after"
    activation_scope: str = "flat"
    version: str = "feature-conv1d-layer-prior-r0-v1"

    def __post_init__(self) -> None:
        if (
            self.version != "feature-conv1d-layer-prior-r0-v1"
            or type(self.geometries) is not tuple
            or not self.geometries
            or any(type(g) is not Conv1DGeometry for g in self.geometries)
            or type(self.activations) is not tuple
            or not self.activations
            or any(type(a) is not CausalActivation for a in self.activations)
            or type(self.kernel_laws) is not tuple
            or not self.kernel_laws
            or any(law not in LAWS for law in self.kernel_laws)
            or self.bias_law not in LAWS
            or any(
                not finite_number(v)
                for v in (
                    self.kernel_scale,
                    self.kernel_mean,
                    self.bias_scale,
                    self.bias_mean,
                    self.student_df,
                )
            )
            or self.kernel_scale < 0
            or self.bias_scale < 0
            or self.student_df <= 0
            or type(self.fan_in) is not bool
            or self.placement not in ("before", "after")
            or self.activation_scope not in ("flat", "channels", "positions")
        ):
            raise ValueError("invalid explicit convolution initialization prior")


@dataclass(frozen=True, slots=True)
class FeatureConv1DRequest:
    input_length: int
    input_channels: int
    layers: tuple[Conv1DLayerPrior, ...]
    layout: str = "channel-major"
    limits: CausalLimits = CausalLimits()
    raw_limits: RawDrawLimits = RawDrawLimits()
    profile: str = "oprior-feature-operation-r0-laws-v1"
    version: str = "feature-conv1d-request-r0-v1"

    def __post_init__(self) -> None:
        if (
            self.version != "feature-conv1d-request-r0-v1"
            or self.profile != "oprior-feature-operation-r0-laws-v1"
            or any(
                type(v) is not int or not 1 <= v < 2**31
                for v in (self.input_length, self.input_channels)
            )
            or type(self.layers) is not tuple
            or not self.layers
            or any(type(layer) is not Conv1DLayerPrior for layer in self.layers)
            or self.layout not in ("channel-major", "position-major")
            or type(self.limits) is not CausalLimits
            or type(self.raw_limits) is not RawDrawLimits
        ):
            raise ValueError("invalid source-qualified convolution sampling request")
        self.admit()

    def admit(self, *, limits: CausalLimits | None = None) -> dict[str, int]:
        from r0.modules.priors.lib.feature_conv_sampling_admission import admit_conv_request

        return admit_conv_request(self, limits or self.limits)


@dataclass(frozen=True, slots=True)
class SampledFeatureConv1D:
    request: FeatureConv1DRequest
    function: FrozenFeatureConv1D
    choices: tuple[tuple[int, int, int], ...]
    raw: tuple[RawDrawRecord, ...]
    initial: tuple[GeneratorSnapshot, ...]
    final: tuple[GeneratorSnapshot, ...]
    version: str = "sampled-feature-conv1d-r0-v1"

    def __post_init__(self) -> None:
        if (
            self.version != "sampled-feature-conv1d-r0-v1"
            or type(self.request) is not FeatureConv1DRequest
            or type(self.function) is not FrozenFeatureConv1D
            or type(self.choices) is not tuple
            or len(self.choices) != len(self.request.layers)
            or any(
                type(c) is not tuple or len(c) != 3 or any(type(v) is not int or v < 0 for v in c)
                for c in self.choices
            )
            or type(self.raw) is not tuple
            or any(type(r) is not RawDrawRecord for r in self.raw)
            or any(
                type(s) is not tuple
                or len(s) != 2
                or any(type(r) is not GeneratorSnapshot for r in s)
                for s in (self.initial, self.final)
            )
        ):
            raise ValueError("invalid sampled feature convolution custody")

    @property
    def input_width(self) -> int:
        return self.function.input_width

    @property
    def output_width(self) -> int:
        return self.function.output_width

    @property
    def parameter_count(self) -> int:
        return self.function.parameter_count

    def validate_replay(self, *, limits: CausalLimits | None = None) -> None:
        from r0.modules.priors.lib.exact_frozen_record import exact_frozen_equal
        from r0.modules.priors.lib.feature_conv_sampling import sample_feature_conv1d

        self.request.admit(limits=limits)
        named, raw = (
            s.restore(byte_limit=self.request.raw_limits.state_bytes) for s in self.initial
        )
        if not exact_frozen_equal(
            self, sample_feature_conv1d(self.request, named_rng=named, raw_rng=raw)
        ):
            raise ValueError("feature convolution differs from exact retained raw replay")
