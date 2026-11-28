"""Logical-variable interventions retain joint score and encoded-state geometry."""

from dataclasses import dataclass, replace

from r0.modules.priors.lib.activation_sampling import finite_number
from r0.modules.priors.lib.native_causal import FrequencyVariableDomain, VariableDomain
from r0.modules.priors.lib.native_vector_records import NativeVectorNode


@dataclass(frozen=True, slots=True)
class NativeVectorHardIntervention:
    node: int
    variable: int
    value: float | None = None
    category: str | None = None
    version: str = "native-vector-hard-intervention-v1"

    def __post_init__(self) -> None:
        if (
            self.version != "native-vector-hard-intervention-v1"
            or any(type(i) is not int or i < 0 for i in (self.node, self.variable))
            or (self.value is None) == (self.category is None)
            or self.value is not None
            and not finite_number(self.value)
            or self.category is not None
            and type(self.category) is not str
        ):
            raise ValueError("vector clamp requires a logical variable and one value or category")


@dataclass(frozen=True, slots=True)
class NativeVectorDomainIntervention:
    node: int
    variable: int
    domain: VariableDomain
    version: str = "native-vector-domain-intervention-v1"

    def __post_init__(self) -> None:
        if (
            self.version != "native-vector-domain-intervention-v1"
            or any(type(i) is not int or i < 0 for i in (self.node, self.variable))
            or type(self.domain) not in (VariableDomain, FrequencyVariableDomain)
        ):
            raise ValueError("vector domain change requires a logical variable and frozen domain")


def validate_variable_change(
    change: NativeVectorHardIntervention | NativeVectorDomainIntervention,
    node: NativeVectorNode,
) -> None:
    if change.variable >= len(node.domains):
        raise ValueError("vector intervention variable is outside the node")
    original = node.domains[change.variable]
    if isinstance(change, NativeVectorHardIntervention):
        if original.kind == "continuous":
            if change.category is not None:
                raise ValueError("continuous vector variable requires a numeric value")
        elif change.category not in original.vocabulary:
            raise ValueError("vector category must name a declared vocabulary value")
    else:
        if (original.kind, original.vocabulary) != (change.domain.kind, change.domain.vocabulary):
            raise ValueError("vector domain change must preserve kind and vocabulary")
        domains = list(node.domains)
        domains[change.variable] = change.domain
        revised = replace(node, domains=tuple(domains))
        if revised.spans != node.spans:
            raise ValueError("vector domain change must preserve score and encoded-state spans")
