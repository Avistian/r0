"""Joint-score native vector nodes with explicit logical-variable and encoding spans."""

from dataclasses import dataclass

from r0.modules.priors.lib.causal_limits import CausalLimits
from r0.modules.priors.lib.frozen_parent_function import FrozenParentFunction
from r0.modules.priors.lib.functional_analysis import (
    ChannelCertificate,
    functional_node_certificates,
)
from r0.modules.priors.lib.functional_causal_node import FunctionalCausalNode
from r0.modules.priors.lib.functional_graph_admission import admit_functional_graph
from r0.modules.priors.lib.linear import LinearRecipe
from r0.modules.priors.lib.native_causal import FrequencyVariableDomain, VariableDomain


@dataclass(frozen=True, slots=True)
class NativeVariableSpan:
    variable: int
    score_start: int
    score_stop: int
    state_start: int
    state_stop: int


def variable_spans(domains: tuple[VariableDomain, ...]) -> tuple[NativeVariableSpan, ...]:
    result: list[NativeVariableSpan] = []
    score = state = 0
    for index, domain in enumerate(domains):
        k = len(domain.vocabulary)
        scores = 1 if domain.kind == "continuous" or domain.emission == "threshold" else k
        encoded = 1 if domain.kind == "continuous" else k
        result.append(NativeVariableSpan(index, score, score + scores, state, state + encoded))
        score += scores
        state += encoded
    return tuple(result)


@dataclass(frozen=True, slots=True)
class NativeVectorNode:
    equation: FunctionalCausalNode
    domains: tuple[VariableDomain, ...]
    version: str = "native-vector-node-v1"

    def __post_init__(self) -> None:
        if (
            self.version != "native-vector-node-v1"
            or type(self.equation) is not FunctionalCausalNode
            or type(self.domains) is not tuple
            or not self.domains
            or any(type(d) not in (VariableDomain, FrequencyVariableDomain) for d in self.domains)
        ):
            raise ValueError("native vector node requires its joint equation and ordered domains")
        if self.equation.function.output_width != self.spans[-1].score_stop:
            raise ValueError("joint native vector score width differs from variable spans")
        if self.equation.pipeline and self.equation.pipeline.converters:
            raise ValueError(
                "native vector domains cannot coexist with numerical readout converters"
            )

    @property
    def spans(self) -> tuple[NativeVariableSpan, ...]:
        return variable_spans(self.domains)

    @property
    def state_width(self) -> int:
        return self.spans[-1].state_stop

    @property
    def value_width(self) -> int:
        return len(self.domains)


@dataclass(frozen=True, slots=True)
class NativeVectorRecipe:
    task: LinearRecipe
    nodes: tuple[NativeVectorNode, ...]
    observed: tuple[tuple[int, int], ...]
    target: tuple[int, int]
    limits: CausalLimits = CausalLimits()
    version: str = "native-vector-causal-v1"

    def __post_init__(self) -> None:
        if (
            self.version != "native-vector-causal-v1"
            or type(self.task) is not LinearRecipe
            or type(self.limits) is not CausalLimits
            or type(self.nodes) is not tuple
            or not 1 <= len(self.nodes) <= self.limits.nodes
        ):
            raise ValueError("native vector recipe requires bounded frozen task/nodes/limits")
        if self.task.noise_ratio or self.task.separation != 1 or self.task.class_offsets:
            raise ValueError(
                "native targets bypass label adapters; adapter controls must be default"
            )
        for index, node in enumerate(self.nodes):
            if type(node) is not NativeVectorNode or any(p >= index for p in node.equation.parents):
                raise ValueError("native vector nodes require parent-before-child order")
            widths = tuple(self.nodes[p].state_width for p in node.equation.parents)
            if widths and node.equation.function.input_width != sum(widths):
                raise ValueError("native vector function inputs differ from encoded parent widths")
            function = node.equation.function
            if isinstance(function, FrozenParentFunction) and (
                not widths or tuple(c.input_width for c in function.children) != widths
            ):
                raise ValueError(
                    "native vector parent composition slices differ from encoded widths"
                )
        if type(self.observed) is not tuple or len(self.observed) != self.task.features:
            raise ValueError("native vector observed references differ from feature count")
        for ref in (*self.observed, self.target):
            if (
                type(ref) is not tuple
                or len(ref) != 2
                or any(type(i) is not int for i in ref)
                or not 0 <= ref[0] < len(self.nodes)
                or not 0 <= ref[1] < self.nodes[ref[0]].value_width
            ):
                raise ValueError("native vector readout requires a valid logical variable")
        if self.target in self.observed:
            raise ValueError("native vector target variable cannot be observed")
        domain = self.nodes[self.target[0]].domains[self.target[1]]
        expected = (
            "regression"
            if domain.kind == "continuous"
            else "binary"
            if len(domain.vocabulary) == 2
            else "multiclass"
        )
        if self.task.task_type != expected or self.task.classes != len(domain.vocabulary):
            raise ValueError("native vector target domain differs from task type/classes")
        self.admit()
        native_vector_certificates(self.nodes)

    def admit(self) -> dict[str, int]:
        widths = tuple(n.state_width for n in self.nodes)
        if max(widths) > self.limits.node_channels or sum(widths) > self.limits.total_channels:
            raise ValueError("native vector encoded state channel limit exceeded")
        target = self.nodes[self.target[0]].domains[self.target[1]]
        report = admit_functional_graph(
            tuple(n.equation for n in self.nodes),
            self.limits,
            rows=self.task.rows,
            metadata=(self.task, self.nodes, self.observed, self.target, self.limits),
            features=self.task.features,
            outputs=max(1, len(target.vocabulary)),
        )
        channels = sum(
            n.state_width + 3 * n.value_width + n.equation.function.output_width for n in self.nodes
        )
        sources = sum(
            (n.equation.root.width if n.equation.root else 0)
            + n.equation.function.output_width
            + n.value_width
            for n in self.nodes
        )
        report["estimated_working_bytes"] += (
            256 * self.task.rows * channels + 256 * sum(widths) * sources
        )
        if report["estimated_working_bytes"] > self.limits.working_bytes:
            raise ValueError("estimated native vector working memory exceeds declared limit")
        return report


def native_vector_certificates(nodes: tuple[NativeVectorNode, ...]) -> None:
    states: list[list[ChannelCertificate]] = []
    for index, node in enumerate(nodes):
        scores, _ = functional_node_certificates(
            node.equation, index, [c for p in node.equation.parents for c in states[p]]
        )
        encoded: list[ChannelCertificate] = []
        for domain, span in zip(node.domains, node.spans, strict=True):
            part = scores[span.score_start : span.score_stop]
            if domain.kind == "continuous":
                encoded.extend(part)
            else:
                sources = frozenset[tuple[int, int]]().union(*(c.sources for c in part))
                if domain.emission == "softmax":
                    sources |= {
                        (
                            index,
                            node.equation.function.output_width
                            + (node.equation.root.width if node.equation.root else 0)
                            + span.variable,
                        )
                    }
                encoded.extend([ChannelCertificate(sources)] * (span.state_stop - span.state_start))
        states.append(encoded)
