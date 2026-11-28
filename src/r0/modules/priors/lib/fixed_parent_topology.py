"""Explicit edge custody when geometry follows an already sampled topology."""

from dataclasses import dataclass

import numpy as np

from r0.modules.priors.lib.causal_limits import CausalLimits
from r0.modules.priors.lib.raw_draw_records import RawDrawRecord

_DEFAULT_LIMITS = CausalLimits()


@dataclass(frozen=True, slots=True)
class FixedParentTopology:
    parents: tuple[tuple[int, ...], ...]
    version: str = "fixed-parent-topology-r0-v1"

    def __post_init__(self) -> None:
        if (
            self.version != "fixed-parent-topology-r0-v1"
            or type(self.parents) is not tuple
            or not self.parents
            or any(
                type(row) is not tuple
                or any(type(p) is not int or not 0 <= p < i for p in row)
                or len(set(row)) != len(row)
                for i, row in enumerate(self.parents)
            )
        ):
            raise ValueError("fixed topology requires ordered acyclic exact parent indices")

    @property
    def nodes(self) -> int:
        return len(self.parents)

    def admit(self, limits: CausalLimits) -> None:
        if type(limits) is not CausalLimits:
            raise ValueError("fixed topology requires caller-owned limits")
        edges = sum(map(len, self.parents))
        if (
            self.nodes > limits.nodes
            or self.nodes > limits.total_channels
            or self.nodes + edges > limits.parameters
            or 256 * (self.nodes + edges + 1) > limits.working_bytes
        ):
            raise ValueError("fixed topology metadata exceeds limits")


@dataclass(frozen=True, slots=True)
class RetainedParentTopology:
    prior: FixedParentTopology
    limits: CausalLimits = CausalLimits()
    version: str = "retained-parent-topology-r0-v1"

    @property
    def parents(self) -> tuple[tuple[int, ...], ...]:
        return self.prior.parents

    @property
    def raw(self) -> tuple[RawDrawRecord, ...]:
        return ()

    def validate(self) -> None:
        if (
            type(self.prior) is not FixedParentTopology
            or self.version != "retained-parent-topology-r0-v1"
        ):
            raise ValueError("retained topology requires its exact fixed prior")
        self.prior.admit(self.limits)

    def replay(self, rng: np.random.Generator) -> "RetainedParentTopology":
        self.validate()
        return self


def retain_parent_topology(
    prior: FixedParentTopology, *, limits: CausalLimits = _DEFAULT_LIMITS
) -> RetainedParentTopology:
    record = RetainedParentTopology(prior, limits)
    record.validate()
    return record
