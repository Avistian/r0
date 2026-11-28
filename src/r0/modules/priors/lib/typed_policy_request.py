"""Typed policy requests retain complete graph envelopes and explicit r0 target laws."""

import math
from dataclasses import dataclass, fields

from r0.modules.priors.lib.causal_intervention_records import (
    CausalBase,
    causal_runtime,
)
from r0.modules.priors.lib.causal_limits import CausalLimits
from r0.modules.priors.lib.functional_causal import FunctionalCausalRecipe
from r0.modules.priors.lib.functional_causal_node import FunctionalCausalNode
from r0.modules.priors.lib.intervention_policy_laws import (
    InterventionParameterLaw,
)
from r0.modules.priors.lib.raw_draw_records import RawDrawLimits
from r0.modules.priors.lib.typed_policy_locations import InterventionPolicyLocation


@dataclass(frozen=True, slots=True)
class TypedInterventionPolicyRequest:
    base: CausalBase
    label_nodes: tuple[int, ...]
    locations: tuple[InterventionPolicyLocation, ...] = ()
    kind: str = "hard"
    eligibility: tuple[int, ...] = ()
    count: int | None = None
    fraction: float = 0.05
    mode: str = "population"
    population: int = 1
    bounds_lifecycle: str = "independent-calibration-r0"
    reference_rows: int = 64
    reference_seed: int = 0
    reference_task_id: int = 0
    parameter_laws: tuple[InterventionParameterLaw, ...] = ()
    limits: CausalLimits = CausalLimits()
    raw_limits: RawDrawLimits = RawDrawLimits()
    profile: str = "explicit-typed-policy-r0-v1"
    version: str = "typed-intervention-policy-request-v1"

    @property
    def graph(self):
        return causal_runtime(self.base)

    @property
    def equations(self) -> tuple[FunctionalCausalNode, ...]:
        graph = self.graph
        if isinstance(graph, FunctionalCausalRecipe):
            return graph.nodes
        return tuple(n.equation for n in graph.nodes)

    @property
    def eligible(self) -> tuple[int, ...]:
        return self.eligibility or tuple(
            i for i in range(len(self.graph.nodes)) if i not in self.label_nodes
        )

    @property
    def selected_count(self) -> int:
        return (
            self.count if self.count is not None else math.ceil(self.fraction * len(self.eligible))
        )

    def __post_init__(self) -> None:
        from r0.modules.priors.lib.typed_policy_validation import validate_typed_policy_values

        validate_typed_policy_values(
            {field.name: getattr(self, field.name) for field in fields(self)}, self.graph
        )
