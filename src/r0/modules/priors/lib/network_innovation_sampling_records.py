"""Recorded construction laws for frozen noise statistics and outgoing importance."""

from dataclasses import dataclass

from r0.modules.priors.lib.generator_snapshot import GeneratorSnapshot
from r0.modules.priors.lib.network_innovation_plan import NetworkInnovationPlan
from r0.modules.priors.lib.network_recipe import NetworkRecipe
from r0.modules.priors.lib.network_scalar_priors import NetworkScalarPrior, NetworkScalarValue
from r0.modules.priors.lib.raw_draw_records import RawDrawLimits, RawDrawRecord


@dataclass(frozen=True, slots=True)
class NetworkInnovationRequest:
    scale: NetworkScalarPrior
    mean: NetworkScalarPrior = NetworkScalarPrior(low=0.0, high=0.0)
    importance: NetworkScalarPrior | None = None
    scale_scope: str = "shared"
    mean_scope: str = "shared"
    channel_scale: str = "shared"
    placement: str = "post-block"
    exempt_first: bool = False
    importance_gates: bool = False
    profile: str = "r0-explicit"
    raw_limits: RawDrawLimits = RawDrawLimits()
    version: str = "network-innovation-request-v1"

    def __post_init__(self) -> None:
        if (
            self.version != "network-innovation-request-v1"
            or type(self.scale) is not NetworkScalarPrior
            or type(self.mean) is not NetworkScalarPrior
            or self.importance is not None
            and type(self.importance) is not NetworkScalarPrior
            or self.scale_scope not in ("shared", "per-block", "per-channel")
            or self.mean_scope not in ("shared", "per-block", "per-channel")
            or self.channel_scale not in ("shared", "half-normal")
            or self.placement not in ("post-block", "pre-activation")
            or type(self.exempt_first) is not bool
            or type(self.importance_gates) is not bool
            or self.importance_gates
            and self.importance is None
            or type(self.raw_limits) is not RawDrawLimits
        ):
            raise ValueError("invalid network innovation sampling request")
        # Profile membership/placement rules are validated through the resulting plan.
        if self.profile not in (
            "r0-explicit",
            "p1-paper-placement",
            "p1-778f0d0-placement",
            "tabicl-9530bd6-placement",
        ):
            raise ValueError("unknown network innovation profile")
        if self.profile in ("p1-778f0d0-placement", "tabicl-9530bd6-placement") and (
            self.mean != NetworkScalarPrior(low=0.0, high=0.0)
            or self.scale_scope != "shared"
            or self.mean_scope != "shared"
            or not self.exempt_first
            or self.placement != "post-block"
        ):
            raise ValueError(
                "pinned noise requires centered shared base scale and first-affine exemption"
            )
        if self.profile == "p1-paper-placement" and self.placement != "pre-activation":
            raise ValueError("paper placement requires pre-activation noise")


@dataclass(frozen=True, slots=True)
class SampledNetworkInnovations:
    request: NetworkInnovationRequest
    plan: NetworkInnovationPlan
    parameters: tuple[NetworkScalarValue, ...]
    raw: tuple[RawDrawRecord, ...]
    initial: GeneratorSnapshot
    final: GeneratorSnapshot
    version: str = "sampled-network-innovations-v1"

    def validate_replay(self, source: NetworkRecipe) -> None:
        from r0.modules.priors.lib.network_innovation_sampling import sample_innovation_record

        result = sample_innovation_record(
            source,
            self.request,
            self.initial.restore(byte_limit=self.request.raw_limits.state_bytes),
        )
        if result != self:
            raise ValueError("network noise statistics differ from complete construction replay")


def network_source_innovations(
    *,
    source: str,
    pre_sample_scales: bool = False,
    scale: NetworkScalarPrior | None = None,
    mean: NetworkScalarPrior | None = None,
    importance: NetworkScalarPrior | None = None,
) -> NetworkInnovationRequest:
    """Placement/scale laws only; undisclosed paper means/importance stay caller-authored."""
    if source not in ("p1-paper", "p1-778f0d0", "tabicl-9530bd6"):
        raise ValueError("unknown neural noise source profile")
    if source != "p1-paper" and mean is not None:
        raise ValueError("pinned implementations have centered noise")
    if source == "p1-paper" and mean is None:
        raise ValueError("paper mean law is undisclosed: provide an explicit adapted mean prior")
    if source == "p1-778f0d0" and scale is None:
        raise ValueError("pinned P1 placement requires an explicit scale law; no paper/code alias")
    family = "tabicl-9530bd6-tnlu-v1" if source == "tabicl-9530bd6" else "p1-paper-tnlu-v1"
    return NetworkInnovationRequest(
        scale or NetworkScalarPrior(family, 0.0001, 0.3),
        mean=mean or NetworkScalarPrior(low=0.0, high=0.0),
        importance=importance,
        channel_scale="half-normal" if pre_sample_scales else "shared",
        placement="pre-activation" if source == "p1-paper" else "post-block",
        exempt_first=source != "p1-paper",
        profile=source + "-placement",
    )
