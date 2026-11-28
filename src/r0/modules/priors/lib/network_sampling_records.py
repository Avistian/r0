"""Versioned standalone architecture requests and replay-bound construction records."""

from dataclasses import dataclass

from r0.modules.priors.lib.activation_sampling import finite_number
from r0.modules.priors.lib.causal_activations import CausalActivation
from r0.modules.priors.lib.causal_limits import CausalLimits
from r0.modules.priors.lib.generator_snapshot import GeneratorSnapshot
from r0.modules.priors.lib.raw_draw_records import RawDrawLimits, RawDrawRecord
from r0.modules.priors.lib.structural_network import FrozenNetworkFunction

FAMILIES = ("dense", "individual-sparse", "block-sparse", "skip", "residual", "irregular", "gated")


@dataclass(frozen=True, slots=True)
class NetworkSamplingRequest:
    input_width: int
    widths: tuple[int, ...]
    families: tuple[str, ...] = FAMILIES
    activations: tuple[CausalActivation, ...] = (
        CausalActivation("tanh"),
        CausalActivation("relu"),
        CausalActivation(),
    )
    homogeneous: bool = False
    placement: str = "after"
    output_activation: bool = False
    weight_laws: tuple[str, ...] = ("normal",)
    weight_scale: float = 1.0
    fan_in: bool = True
    bias_scale: float = 0.2
    dropout: float = 0.2
    dropout_profile: str = "fixed"
    mask_profile: str = "generic"
    scale_exponent: float = 0.5
    block_limit: int = 2
    experts: int = 3
    skip_probability: float = 0.5
    limits: CausalLimits = CausalLimits()
    raw_limits: RawDrawLimits = RawDrawLimits()
    version: str = "network-sampling-request-v1"

    def __post_init__(self) -> None:
        if (
            self.version != "network-sampling-request-v1"
            or type(self.input_width) is not int
            or self.input_width < 1
            or type(self.widths) is not tuple
            or not self.widths
            or any(type(w) is not int or w < 1 for w in self.widths)
            or type(self.families) is not tuple
            or not self.families
            or any(f not in FAMILIES for f in self.families)
            or type(self.activations) is not tuple
            or not self.activations
            or any(type(a) is not CausalActivation for a in self.activations)
            or type(self.weight_laws) is not tuple
            or not self.weight_laws
            or any(
                law not in ("normal", "uniform", "student_t", "cauchy") for law in self.weight_laws
            )
            or self.placement not in ("before", "after")
            or any(
                type(v) is not bool for v in (self.homogeneous, self.output_activation, self.fan_in)
            )
            or any(
                not finite_number(v) or v < 0
                for v in (self.weight_scale, self.bias_scale, self.dropout, self.skip_probability)
            )
            or self.dropout > 1
            or self.skip_probability > 1
            or self.dropout_profile not in ("fixed", "p1-beta")
            or self.mask_profile
            not in ("generic", "p1-individual", "p1-block", "tabicl-individual", "tabicl-block")
            or self.scale_exponent
            not in ((0.0, 0.5) if self.mask_profile.startswith("tabicl-") else (0.5, 1.0))
            or any(type(v) is not int or v < 1 for v in (self.block_limit, self.experts))
            or type(self.limits) is not CausalLimits
            or type(self.raw_limits) is not RawDrawLimits
        ):
            raise ValueError(
                "network request requires explicit valid architecture, laws and limits"
            )
        if self.mask_profile != "generic" and (
            self.families
            != (
                ("individual-sparse",)
                if self.mask_profile.endswith("individual")
                else ("block-sparse",)
            )
            or self.weight_laws != ("normal",)
            or self.fan_in
        ):
            raise ValueError(
                "Source mask profiles require their matching sparse family, Gaussian weights "
                "and no generic fan-in scaling"
            )
        if self.dropout_profile == "p1-beta" and any(
            f != "individual-sparse" for f in self.families
        ):
            raise ValueError("P1 Beta dropout applies to individual edge masks only")
        self.admit()

    def admit(self) -> dict[str, int]:
        widths = (self.input_width, *self.widths)
        # Worst-case all earlier parents, all skips, every expert and explicit masks.
        numbers, draws = network_construction_cost(self)
        rawbytes = 1024 * draws + 512 * numbers + 2 * self.raw_limits.state_bytes
        if (
            numbers > self.raw_limits.numbers
            or draws > self.raw_limits.draws
            or rawbytes > self.raw_limits.working_bytes
        ):
            raise ValueError("network raw construction worst-case limit exceeded")
        result = self.limits.admit_dimensions(
            (*widths, self.experts),
            tuple(() for _ in (*widths, self.experts)),
            rows=16,
            calibration_rows=0,
            features=self.input_width,
            outputs=self.widths[-1],
            extra_parameters=4 * numbers,
            extra_working_channels=32 * sum(widths) * self.experts,
        )
        if result["estimated_working_bytes"] + rawbytes > self.limits.working_bytes:
            raise ValueError("network combined construction working limit exceeded")
        return result


@dataclass(frozen=True, slots=True)
class SampledNetwork:
    request: NetworkSamplingRequest
    function: FrozenNetworkFunction
    choices: tuple[tuple[str, str], ...]
    raw: tuple[RawDrawRecord, ...]
    initial: tuple[GeneratorSnapshot, ...]
    final: tuple[GeneratorSnapshot, ...]
    version: str = "sampled-structural-network-v1"

    def validate_replay(self) -> None:
        from r0.modules.priors.lib.network_sampling import sample_network

        if self.version != "sampled-structural-network-v1" or len(self.initial) != 2:
            raise ValueError("sampled network requires its version and two initial streams")
        named, raw = (
            s.restore(byte_limit=self.request.raw_limits.state_bytes) for s in self.initial
        )
        if sample_network(self.request, named_rng=named, raw_rng=raw) != self:
            raise ValueError("network construction differs from retained complete replay")


def network_construction_cost(request: NetworkSamplingRequest) -> tuple[int, int]:
    """Worst-case raw matrix/mask numbers and draw records before allocation."""
    widths = (request.input_width, *request.widths)
    numbers = 0
    for i, out in enumerate(request.widths):
        incoming = sum(widths[: i + 1])
        numbers += (
            request.experts * (2 * incoming * out + out)
            + incoming * request.experts
            + request.experts
        )
        numbers += incoming * out + (i + 1) * out
    draws = 32 + len(request.widths) * (16 * request.experts + 8 * len(widths))
    return numbers, draws
