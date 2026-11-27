"""Typed policy requests retain complete graph envelopes and explicit r0 target laws."""

import math
from dataclasses import dataclass

from r0.modules.priors.lib.causal_intervention_records import (
    CausalBase,
    CausalInterventionRecipe,
    SoftIntervention,
    causal_runtime,
)
from r0.modules.priors.lib.causal_limits import CausalLimits
from r0.modules.priors.lib.converted_causal_records import ConvertedCausalRecipe
from r0.modules.priors.lib.functional_causal import FunctionalCausalRecipe
from r0.modules.priors.lib.functional_causal_node import FunctionalCausalNode
from r0.modules.priors.lib.intervention_policy_laws import (
    InterventionParameterLaw,
    numeric_leaves,
    parameter_group,
)
from r0.modules.priors.lib.native_causal import NativeCausalRecipe
from r0.modules.priors.lib.native_vector_records import NativeVectorRecipe
from r0.modules.priors.lib.raw_draw_records import RawDrawLimits
from r0.modules.priors.lib.typed_policy_locations import InterventionPolicyLocation, PolicyClamp


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
        graph = self.graph
        if self.base.version in (
            "support-population-functional-causal-v1",
            "population-fitted-functional-causal-v1",
            "sampled-population-fitted-functional-causal-v1",
        ):
            raise ValueError(
                "population/support refitting policies remain separate from frozen typed policies"
            )
        if isinstance(graph, ConvertedCausalRecipe):
            targets = {graph.target.node}
        elif isinstance(graph, (NativeCausalRecipe, NativeVectorRecipe)):
            targets = {graph.target[0]}
        else:
            targets = {i for i, _ in graph.target}
        if (
            self.version != "typed-intervention-policy-request-v1"
            or self.profile != "explicit-typed-policy-r0-v1"
            or self.kind not in ("hard", "soft")
            or self.mode not in ("matched", "population")
            or self.bounds_lifecycle
            not in ("independent-calibration-r0", "source-reference-population")
            or type(self.limits) is not CausalLimits
            or type(self.raw_limits) is not RawDrawLimits
            or type(self.fraction) not in (int, float)
            or not math.isfinite(self.fraction)
            or not 0 < self.fraction <= 1
            or self.count is not None
            and (type(self.count) is not int or self.count < 1)
            or any(
                type(v) is not int or not 0 <= v < 2**63
                for v in (self.reference_seed, self.reference_task_id)
            )
            or type(self.population) is not int
            or not 1 <= self.population < 2**63
            or type(self.reference_rows) is not int
            or not 1 <= self.reference_rows <= 20000
        ):
            raise ValueError("invalid explicit typed intervention policy")
        for values in (self.label_nodes, self.eligibility):
            if (
                type(values) is not tuple
                or len(set(values)) != len(values)
                or any(type(i) is not int or not 0 <= i < len(graph.nodes) for i in values)
            ):
                raise ValueError("typed policy node indices must be unique exact integers")
        if (
            set(self.label_nodes) != targets
            or self.kind == "soft"
            and bool(set(self.eligible) & targets)
            or not self.eligible
            or self.selected_count > len(self.eligible)
        ):
            raise ValueError(
                "typed policy cannot replace label equations or silently repair eligibility/count"
            )
        if (
            self.bounds_lifecycle == "source-reference-population"
            and self.reference_rows != graph.task.rows
        ):
            raise ValueError("source reference requires the complete original population")
        if type(self.locations) is not tuple or any(
            type(x) is not InterventionPolicyLocation for x in self.locations
        ):
            raise ValueError("typed policy locations must be frozen explicit records")
        if type(self.parameter_laws) is not tuple or any(
            type(x) is not InterventionParameterLaw for x in self.parameter_laws
        ):
            raise ValueError("typed parameter laws must be frozen explicit records")
        if self.kind == "hard":
            if self.parameter_laws or {x.node for x in self.locations} != set(self.eligible):
                raise ValueError(
                    "every eligible hard node requires explicit locations and no soft laws"
                )
            changes: list[PolicyClamp] = []
            for address in self.locations:
                if address.node in targets:
                    if isinstance(graph, ConvertedCausalRecipe):
                        overlaps = (
                            address.kind == "capture"
                            and address.capture == graph.target.capture
                            or address.kind == "numerical"
                            and graph.target.channel in address.channels
                        )
                    elif isinstance(graph, (NativeCausalRecipe, NativeVectorRecipe)):
                        overlaps = address.variable == graph.target[1]
                    else:
                        overlaps = any(
                            node == address.node and channel in address.channels
                            for node, channel in graph.target
                        )
                    if overlaps:
                        raise ValueError(
                            "typed policy cannot directly clamp a declared target variable"
                        )
                vocabulary = address.vocabulary(graph.nodes[address.node])
                values = () if vocabulary else (0.0,) * (len(address.channels) or 1)
                changes.append(
                    address.clamp(
                        graph.nodes[address.node], values, vocabulary[0] if vocabulary else None
                    )
                )
            CausalInterventionRecipe(self.base, tuple(changes), self.mode, self.population)
        else:
            if self.locations or {x.node for x in self.parameter_laws} != set(self.eligible):
                raise ValueError(
                    "every eligible soft equation requires explicit laws and no hard locations"
                )
            seen: list[tuple[int, tuple[str, ...]]] = []
            for law in self.parameter_laws:
                if any(
                    n == law.node
                    and (path[: len(law.path)] == law.path or law.path[: len(path)] == path)
                    for n, path in seen
                ):
                    raise ValueError("overlapping typed parameter paths")
                try:
                    numeric_leaves(parameter_group(self.equations[law.node].function, law.path)[2])
                except (KeyError, IndexError, AttributeError, TypeError) as error:
                    raise ValueError("invalid typed parameter path") from error
                seen.append((law.node, law.path))
            CausalInterventionRecipe(
                self.base,
                tuple(
                    SoftIntervention(i, function=self.equations[i].function) for i in self.eligible
                ),
                self.mode,
                self.population,
            )
