"""Fixed-size source topology laws for the typed multi-parent envelope."""

from dataclasses import dataclass

from r0.modules.priors.lib.cauchy_graph import CauchyGraphPrior
from r0.modules.priors.lib.causal_limits import CausalLimits
from r0.modules.priors.lib.geometry_topology import geometry_topology_raw_reservation
from r0.modules.priors.lib.raw_draw_records import RawDrawLimits
from r0.modules.priors.lib.structural_graph_prior import (
    AttachmentGraphPrior,
    MotifGraphPrior,
    admit_structural_graph,
)


@dataclass(frozen=True, slots=True)
class StructuralTopologyRequest:
    prior: CauchyGraphPrior | AttachmentGraphPrior | MotifGraphPrior
    version: str = "structural-topology-request-r0-v1"

    def __post_init__(self) -> None:
        if self.version != "structural-topology-request-r0-v1" or type(self.prior) not in (
            CauchyGraphPrior,
            AttachmentGraphPrior,
            MotifGraphPrior,
        ):
            raise ValueError("structural topology requires a complete supported source prior")

    @property
    def minimum_nodes(self) -> int:
        return self.prior.nodes

    @property
    def maximum_nodes(self) -> int:
        return self.prior.nodes

    def admit(self, limits: CausalLimits, raw_limits: RawDrawLimits) -> dict[str, int]:
        n = self.prior.nodes
        if n > min(limits.nodes, limits.total_channels):
            raise ValueError("structural topology node limit exceeded")
        if not isinstance(self.prior, CauchyGraphPrior):
            admit_structural_graph(self.prior, limits)
        entries = (
            n * n * min(self.prior.edges_per_node, n)
            if isinstance(self.prior, AttachmentGraphPrior)
            else n * n * (self.prior.rewrite_steps + self.prior.depth + 1)
            if isinstance(self.prior, MotifGraphPrior)
            else n * n
        )
        draws, numbers, extra = geometry_topology_raw_reservation(self.prior)
        parameters = 4 * entries + 16 * n + 512 + extra
        memory = 512 * (entries + 4 * n + 1) + 8 * 1024**2 + extra
        raw_memory = 128 * numbers + 16384 * draws + extra
        memory += raw_memory
        if (
            parameters > limits.parameters
            or memory > limits.working_bytes
            or draws > raw_limits.draws
            or numbers > raw_limits.numbers
            or raw_memory > raw_limits.working_bytes
        ):
            raise ValueError("structural topology complete source support exceeds limits")
        return dict(
            parameters=parameters,
            estimated_working_bytes=memory,
            raw_numbers=numbers,
            raw_draws=draws,
            raw_working_bytes=raw_memory,
        )
