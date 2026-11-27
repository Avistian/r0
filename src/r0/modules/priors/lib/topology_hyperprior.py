"""Explicit r0 reconstruction of log-size/Gamma redirection hyperpriors.

The source does not disclose integer rounding or how an unbounded Gamma becomes
probability. These policies are named r0 choices, not exact source reproduction.
"""

import math
from dataclasses import dataclass

from r0.modules.priors.lib.activation_sampling import finite_number
from r0.modules.priors.lib.causal_graph import CausalGraphPrior


@dataclass(frozen=True, slots=True)
class TopologyHyperprior:
    log_nodes_low: float = math.log(4)
    log_nodes_high: float = math.log(32)
    rounding: str = "nearest-even"
    gamma_shape: float = 2.0
    gamma_rate: float = 2.0
    components: int = 1
    probability_transform: str = "gamma-odds-r0"
    version: str = "topology-hyperprior-r0-v1"

    def __post_init__(self) -> None:
        if (
            self.version != "topology-hyperprior-r0-v1"
            or self.probability_transform not in ("gamma-odds-r0", "gamma-exp-r0")
            or self.rounding not in ("floor", "ceil", "nearest-even")
            or any(
                not finite_number(v)
                for v in (
                    self.log_nodes_low,
                    self.log_nodes_high,
                    self.gamma_shape,
                    self.gamma_rate,
                )
            )
            or not 0 <= self.log_nodes_low < self.log_nodes_high <= math.log(1_000_000)
            or self.gamma_shape <= 0
            or self.gamma_rate <= 0
            or type(self.components) is not int
            or not 1 <= self.components <= self.node_count(self.log_nodes_low)
        ):
            raise ValueError("invalid explicit topology hyperprior laws or component bounds")

    def node_count(self, log_nodes: float) -> int:
        value = math.exp(log_nodes)
        if self.rounding == "floor":
            return math.floor(value)
        if self.rounding == "ceil":
            return math.ceil(value)
        return round(value)

    @property
    def minimum_nodes(self) -> int:
        return self.node_count(self.log_nodes_low)

    @property
    def maximum_nodes(self) -> int:
        return self.node_count(self.log_nodes_high)


@dataclass(frozen=True, slots=True)
class FixedSizeTopologyHyperprior:
    nodes: int
    gamma_shape: float = 2.0
    gamma_rate: float = 2.0
    components: int = 1
    probability_transform: str = "gamma-odds-r0"
    version: str = "fixed-size-topology-hyperprior-r0-v1"

    def __post_init__(self) -> None:
        if (
            self.version != "fixed-size-topology-hyperprior-r0-v1"
            or self.probability_transform not in ("gamma-odds-r0", "gamma-exp-r0")
            or type(self.nodes) is not int
            or not 1 <= self.nodes <= 1_000_000
            or type(self.components) is not int
            or not 1 <= self.components <= self.nodes
            or not finite_number(self.gamma_shape)
            or self.gamma_shape <= 0
            or not finite_number(self.gamma_rate)
            or self.gamma_rate <= 0
        ):
            raise ValueError("fixed-size topology hyperprior requires explicit finite laws")

    @property
    def minimum_nodes(self) -> int:
        return self.nodes

    @property
    def maximum_nodes(self) -> int:
        return self.nodes


type TopologyHyperpriorRequest = TopologyHyperprior | FixedSizeTopologyHyperprior


def transform_topology_hyperprior(
    request: TopologyHyperpriorRequest, *, log_nodes: float | None, gamma_unit: float
) -> CausalGraphPrior:
    """Apply retained log-size and unit-scale Gamma draws without clipping/retries."""
    if type(request) not in (TopologyHyperprior, FixedSizeTopologyHyperprior):
        raise ValueError("topology transformation requires a versioned explicit hyperprior")
    if isinstance(request, TopologyHyperprior):
        if (
            log_nodes is None
            or not finite_number(log_nodes)
            or not request.log_nodes_low <= log_nodes <= request.log_nodes_high
        ):
            raise ValueError("log-size draw is outside its declared support")
        nodes = request.node_count(log_nodes)
    else:
        if log_nodes is not None:
            raise ValueError("fixed-size topology must not retain an unused log-size draw")
        nodes = request.nodes
    if not finite_number(gamma_unit) or gamma_unit <= 0:
        raise ValueError("Gamma draw must be finite positive; underflow is not retried")
    gamma = gamma_unit / request.gamma_rate
    if not math.isfinite(gamma) or gamma <= 0:
        raise ValueError("nonfinite scaled Gamma draw; no clipping or retry")
    # The alternative expression avoids overflowing 1+gamma at extreme scales.
    probability = gamma / (1 + gamma) if gamma <= 1 else 1 / (1 + 1 / gamma)
    if request.probability_transform == "gamma-exp-r0":
        probability = -math.expm1(-gamma)
    return CausalGraphPrior(
        family="redirection",
        nodes=nodes,
        components=request.components,
        redirection_probability=probability,
    )
