"""Opt-in source hyperpriors with explicit operational admission, not source clipping."""

from dataclasses import dataclass, replace

from r0.modules.priors.lib.generator_snapshot import GeneratorSnapshot
from r0.modules.priors.lib.joint_function_admission import function_record_cost
from r0.modules.priors.lib.network_sampling_records import (
    NetworkSamplingRequest,
    SampledNetwork,
    network_construction_cost,
)
from r0.modules.priors.lib.network_scalar_priors import NetworkScalarPrior, NetworkScalarValue
from r0.modules.priors.lib.raw_draw_records import RawDrawRecord


@dataclass(frozen=True, slots=True)
class NetworkHyperpriorRequest:
    template: NetworkSamplingRequest
    depth: NetworkScalarPrior = NetworkScalarPrior("p1-paper-tnlu-v1", 1.0, 6.0, 2.0, True)
    width: NetworkScalarPrior = NetworkScalarPrior("p1-paper-tnlu-v1", 5.0, 130.0, 4.0, True)
    weight_scale: NetworkScalarPrior = NetworkScalarPrior("p1-paper-tnlu-v1", 0.01, 10.0)
    bias_scale: NetworkScalarPrior | None = None
    width_scope: str = "shared"
    depth_semantics: str = "hidden-blocks-plus-output"
    max_hidden_blocks: int = 8
    max_hidden_width: int = 32
    version: str = "network-hyperprior-request-v1"

    def __post_init__(self) -> None:
        if (
            self.version != "network-hyperprior-request-v1"
            or type(self.template) is not NetworkSamplingRequest
            or any(
                type(v) is not NetworkScalarPrior
                for v in (self.depth, self.width, self.weight_scale)
            )
            or self.bias_scale is not None
            and type(self.bias_scale) is not NetworkScalarPrior
            or self.width_scope not in ("shared", "per-hidden-block")
            or self.depth_semantics not in ("hidden-blocks-plus-output", "total-blocks")
            or type(self.max_hidden_blocks) is not int
            or not 1 <= self.max_hidden_blocks <= 64
            or type(self.max_hidden_width) is not int
            or not 1 <= self.max_hidden_width <= 4096
            or len(self.template.widths) != 1
            or self.template.weight_scale != 1.0
        ):
            raise ValueError(
                "network hyperprior requires a fixed-output template and explicit laws/caps"
            )
        if self.bias_scale is not None and self.template.mask_profile.startswith(
            ("p1-", "tabicl-")
        ):
            raise ValueError(
                "Source mask profiles use uniform biases; a bias-scale draw would be unused"
            )
        self.admit()

    def admit(self) -> None:
        metadata, storage = function_record_cost(self, self.template.limits)
        # Bounded cap tuple, never a source realization or silently truncated draw.
        maximum = replace(
            self.template,
            widths=(self.max_hidden_width,) * self.max_hidden_blocks + self.template.widths,
        )
        report = maximum.admit()
        numbers, draws = network_construction_cost(maximum)
        rawbytes = 1024 * draws + 512 * numbers + 2 * self.template.raw_limits.state_bytes
        scalar_count = self.max_hidden_blocks + 3
        extra = 16384 * scalar_count + 4 * self.template.raw_limits.state_bytes
        if (
            report["estimated_working_bytes"] + rawbytes + extra + 4 * storage
            > self.template.limits.working_bytes
        ):
            raise ValueError("hyperprior envelope working limit exceeded before draws")
        if (
            scalar_count * 8 + draws > self.template.raw_limits.draws
            or scalar_count * 8 + numbers > self.template.raw_limits.numbers
            or rawbytes + extra > self.template.raw_limits.working_bytes
            or report["parameters"] + scalar_count * 256 + metadata
            > self.template.limits.parameters
        ):
            raise ValueError("hyperprior scalar raw budget exceeded before draws")


@dataclass(frozen=True, slots=True, kw_only=True)
class SampledHyperpriorNetwork(SampledNetwork):
    profile: NetworkHyperpriorRequest
    parameters: tuple[NetworkScalarValue, ...]
    hyper_raw: tuple[RawDrawRecord, ...]
    hyper_initial: GeneratorSnapshot
    hyper_final: GeneratorSnapshot
    version: str = "sampled-hyperprior-network-v1"

    def validate_replay(self) -> None:
        from r0.modules.priors.lib.network_hyperprior_sampling import sample_hyperprior_network

        if self.version != "sampled-hyperprior-network-v1" or len(self.initial) != 2:
            raise ValueError("profiled network requires original stream boundaries and version")
        result = sample_hyperprior_network(
            self.profile,
            hyper_rng=self.hyper_initial.restore(
                byte_limit=self.profile.template.raw_limits.state_bytes
            ),
            named_rng=self.initial[0].restore(
                byte_limit=self.profile.template.raw_limits.state_bytes
            ),
            raw_rng=self.initial[1].restore(
                byte_limit=self.profile.template.raw_limits.state_bytes
            ),
        )
        if result != self:
            raise ValueError("profiled network differs from full scalar and architecture replay")


def network_source_hyperprior(
    template: NetworkSamplingRequest,
    *,
    source: str,
    max_hidden_blocks: int = 8,
    max_hidden_width: int = 32,
) -> NetworkHyperpriorRequest:
    """Source scalar laws on an explicit r0 architecture template, not full prior reproduction."""
    if source not in ("p1-paper", "tabicl-9530bd6"):
        raise ValueError("unknown source scalar-law profile")
    family = source + "-tnlu-v1"
    return NetworkHyperpriorRequest(
        template,
        depth=NetworkScalarPrior(family, 1.0, 6.0, 2.0, True),
        width=NetworkScalarPrior(family, 5.0, 130.0, 4.0, True),
        weight_scale=NetworkScalarPrior(family, 0.01, 10.0),
        max_hidden_blocks=max_hidden_blocks,
        max_hidden_width=max_hidden_width,
    )
