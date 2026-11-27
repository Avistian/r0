"""Explicit typed intervention addresses and declared categorical sampling laws."""

import math
from dataclasses import dataclass

from r0.modules.priors.lib.causal_intervention_records import HardIntervention
from r0.modules.priors.lib.converted_causal_records import ConvertedCausalNode
from r0.modules.priors.lib.converter_intervention import ConverterHardIntervention
from r0.modules.priors.lib.functional_causal_node import FunctionalCausalNode
from r0.modules.priors.lib.native_causal import NativeCausalNode
from r0.modules.priors.lib.native_vector_intervention import NativeVectorHardIntervention
from r0.modules.priors.lib.native_vector_records import NativeVectorNode

type PolicyNode = FunctionalCausalNode | NativeCausalNode | NativeVectorNode | ConvertedCausalNode
type PolicyClamp = HardIntervention | NativeVectorHardIntervention | ConverterHardIntervention


@dataclass(frozen=True, slots=True)
class InterventionPolicyLocation:
    node: int
    kind: str = "numerical"
    channels: tuple[int, ...] = ()
    variable: int | None = None
    capture: str | None = None
    location: str = "state_and_readout"
    probabilities: tuple[float, ...] = ()
    version: str = "intervention-policy-location-r0-v1"

    def __post_init__(self) -> None:
        if (
            self.version != "intervention-policy-location-r0-v1"
            or type(self.node) is not int
            or self.node < 0
            or self.kind not in ("numerical", "native-variable", "capture")
            or type(self.channels) is not tuple
            or any(type(c) is not int or c < 0 for c in self.channels)
            or len(set(self.channels)) != len(self.channels)
            or self.variable is not None
            and (type(self.variable) is not int or self.variable < 0)
            or self.capture is not None
            and (type(self.capture) is not str or not self.capture)
            or type(self.probabilities) is not tuple
            or any(
                type(p) not in (int, float) or not math.isfinite(p) or p < 0
                for p in self.probabilities
            )
            or self.probabilities
            and not math.isclose(sum(self.probabilities), 1.0, rel_tol=0, abs_tol=1e-12)
        ):
            raise ValueError("invalid typed policy location or categorical probabilities")
        if self.kind == "numerical":
            valid = (
                bool(self.channels)
                and self.variable is None
                and self.capture is None
                and not self.probabilities
                and self.location in ("state", "readout", "state_and_readout")
            )
        elif self.kind == "native-variable":
            valid = (
                not self.channels
                and self.variable is not None
                and self.capture is None
                and self.location == "native-value"
            )
        else:
            valid = (
                not self.channels
                and self.variable is None
                and self.capture is not None
                and self.location in ("capture", "representation_and_capture")
            )
        if not valid:
            raise ValueError(
                "typed policy requires one explicit numerical, native or capture address"
            )

    def vocabulary(self, node: PolicyNode) -> tuple[str, ...]:
        if self.kind == "native-variable":
            if isinstance(node, NativeCausalNode) and self.variable == 0:
                domain = node.domain
            elif (
                isinstance(node, NativeVectorNode)
                and self.variable is not None
                and self.variable < len(node.domains)
            ):
                domain = node.domains[self.variable]
            else:
                raise ValueError("native policy location requires a valid logical variable")
            vocabulary = domain.vocabulary if domain.kind != "continuous" else ()
        elif self.kind == "capture":
            if not isinstance(node, ConvertedCausalNode):
                raise ValueError("capture policy location requires a converted node")
            converter = next((c for c in node.converters if c.name == self.capture), None)
            if converter is None:
                raise ValueError("policy capture name is not declared")
            vocabulary = converter.vocabulary
        else:
            if isinstance(node, (NativeCausalNode, NativeVectorNode)):
                raise ValueError("native clamps must name logical variables, not score channels")
            equation = node.equation if isinstance(node, ConvertedCausalNode) else node
            if any(c >= equation.function.output_width for c in self.channels):
                raise ValueError("policy channel exceeds equation width")
            vocabulary = ()
        if self.probabilities and len(self.probabilities) != len(vocabulary):
            raise ValueError("policy probabilities must match the full declared vocabulary")
        return vocabulary

    def clamp(
        self, node: PolicyNode, values: tuple[float, ...] = (), category: str | None = None
    ) -> PolicyClamp:
        if self.kind == "capture":
            assert self.capture is not None
            assert category is not None
            return ConverterHardIntervention(self.node, self.capture, category, self.location)
        if isinstance(node, NativeVectorNode):
            assert self.variable is not None
            return NativeVectorHardIntervention(
                self.node, self.variable, values[0] if values else None, category
            )
        if isinstance(node, NativeCausalNode):
            return HardIntervention(self.node, (0,) if values else (), values, category, "state")
        return HardIntervention(self.node, self.channels, values, location=self.location)
