"""Admitted frozen ancestors for fresh downstream fitting without task readouts."""

from dataclasses import dataclass

from r0.modules.priors.lib.causal_limits import CausalLimits
from r0.modules.priors.lib.functional_analysis import functional_certificates
from r0.modules.priors.lib.functional_causal_node import FunctionalCausalNode
from r0.modules.priors.lib.functional_graph_admission import (
    admit_functional_graph,
    validate_functional_nodes,
)


@dataclass(frozen=True, slots=True)
class FunctionalPrefix:
    nodes: tuple[FunctionalCausalNode, ...]
    rows: int
    limits: CausalLimits = CausalLimits()
    retained: tuple[object, ...] = ()

    def __post_init__(self) -> None:
        if type(self.rows) is not int or not 1 <= self.rows < 2**63:
            raise ValueError("functional prefix rows must be positive bounded integers")
        if type(self.retained) is not tuple:
            raise ValueError("functional prefix retained construction metadata must be a tuple")
        validate_functional_nodes(self.nodes, self.limits)
        self.admit()
        functional_certificates(self.nodes)

    def admit(self) -> dict[str, int]:
        """Include external construction metadata retained while this prefix evaluates."""
        return admit_functional_graph(
            self.nodes,
            self.limits,
            rows=self.rows,
            metadata=(self.nodes, self.rows, self.limits, self.retained),
        )
