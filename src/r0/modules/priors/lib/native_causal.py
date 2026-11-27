"""Typed native graph variables; numerical equations emit declared discrete domains."""

import math
from dataclasses import dataclass
from typing import Literal

from r0.modules.priors.lib.causal_limits import CausalLimits
from r0.modules.priors.lib.functional_analysis import (
    ChannelCertificate,
    functional_node_certificates,
)
from r0.modules.priors.lib.functional_causal_node import FunctionalCausalNode
from r0.modules.priors.lib.functional_graph_admission import admit_functional_graph
from r0.modules.priors.lib.linear import LinearRecipe


@dataclass(frozen=True, slots=True)
class VariableDomain:
    kind: Literal["continuous", "binary", "categorical"] = "continuous"
    vocabulary: tuple[str, ...] = ()
    emission: Literal["identity", "softmax", "argmax", "threshold"] = "identity"
    thresholds: tuple[float, ...] = ()
    temperature: float = 1.0
    version: str = "native-variable-domain-v1"

    def __post_init__(self) -> None:
        if self.version != (
            "native-frequency-domain-v2"
            if type(self) is FrequencyVariableDomain
            else "native-variable-domain-v1"
        ) or self.kind not in (
            "continuous",
            "binary",
            "categorical",
        ):
            raise ValueError("unsupported native variable domain")
        if (
            type(self.vocabulary) is not tuple
            or any(type(v) is not str or not v for v in self.vocabulary)
            or len(set(self.vocabulary)) != len(self.vocabulary)
        ):
            raise ValueError("native vocabulary must contain unique nonempty declared labels")
        if self.kind == "continuous":
            if self.vocabulary or self.emission != "identity" or self.thresholds:
                raise ValueError(
                    "continuous variables require identity emission without vocabulary"
                )
        elif (
            len(self.vocabulary) < 2
            or self.kind == "binary"
            and len(self.vocabulary) != 2
            or self.emission not in ("softmax", "argmax", "threshold")
        ):
            raise ValueError("native discrete vocabulary/emission does not match its domain")
        if (
            type(self.thresholds) is not tuple
            or any(type(t) not in (int, float) or not math.isfinite(t) for t in self.thresholds)
            or any(a >= b for a, b in zip(self.thresholds, self.thresholds[1:], strict=False))
        ):
            raise ValueError("thresholds must be finite and strictly increasing")
        if (
            self.emission == "threshold"
            and len(self.thresholds) != len(self.vocabulary) - 1
            or self.emission != "threshold"
            and self.thresholds
        ):
            raise ValueError("threshold emission needs exactly cardinality minus one cutpoints")
        if (
            type(self.temperature) not in (int, float)
            or not math.isfinite(self.temperature)
            or (self.temperature <= 0 or self.emission != "softmax" and self.temperature != 1)
        ):
            raise ValueError("positive temperature applies only to stochastic softmax")


@dataclass(frozen=True, slots=True)
class FrequencyVariableDomain(VariableDomain):
    """Adaptation: positive frequency offsets influence logits, not guaranteed marginals."""

    frequencies: tuple[float, ...] = ()
    version: str = "native-frequency-domain-v2"

    def __post_init__(self) -> None:
        VariableDomain.__post_init__(self)
        if (
            self.kind == "continuous"
            or type(self.frequencies) is not tuple
            or (
                len(self.frequencies) != len(self.vocabulary)
                or any(
                    type(p) not in (int, float) or not math.isfinite(p) or p <= 0
                    for p in self.frequencies
                )
                or not math.isclose(math.fsum(self.frequencies), 1.0, rel_tol=1e-12, abs_tol=1e-12)
            )
        ):
            raise ValueError(
                "frequency domain requires strictly positive normalized category weights"
            )


@dataclass(frozen=True, slots=True)
class NativeCausalNode:
    equation: FunctionalCausalNode
    domain: VariableDomain = VariableDomain()
    version: str = "native-causal-node-v1"

    def __post_init__(self) -> None:
        if (
            self.version != "native-causal-node-v1"
            or type(self.equation) is not FunctionalCausalNode
            or (type(self.domain) not in (VariableDomain, FrequencyVariableDomain))
        ):
            raise ValueError("native node requires a frozen equation and variable domain")
        if self.domain.kind != "continuous" and self.equation.function.output_width != (
            1 if self.domain.emission == "threshold" else len(self.domain.vocabulary)
        ):
            raise ValueError("native score width must match the declared emission")
        if self.equation.pipeline and self.equation.pipeline.converters:
            raise ValueError(
                "native nodes use domain emissions; numerical readout converters conflict"
            )

    @property
    def state_width(self) -> int:
        """Width consumed by descendants: continuous channels or explicit one-hot categories."""
        return (
            self.equation.function.output_width
            if self.domain.kind == "continuous"
            else len(self.domain.vocabulary)
        )

    @property
    def value_width(self) -> int:
        return self.equation.function.output_width if self.domain.kind == "continuous" else 1


@dataclass(frozen=True, slots=True)
class NativeCausalRecipe:
    task: LinearRecipe
    nodes: tuple[NativeCausalNode, ...]
    observed: tuple[tuple[int, int], ...]
    target: tuple[int, int]
    limits: CausalLimits = CausalLimits()
    version: str = "native-causal-v1"

    def __post_init__(self) -> None:
        if (
            self.version != "native-causal-v1"
            or type(self.task) is not LinearRecipe
            or (type(self.limits) is not CausalLimits)
        ):
            raise ValueError("native recipe requires its version, task controls and limits")
        if self.task.noise_ratio or self.task.separation != 1 or self.task.class_offsets:
            raise ValueError(
                "native targets bypass label adapters; adapter controls must be default"
            )
        if type(self.nodes) is not tuple or not self.nodes or len(self.nodes) > self.limits.nodes:
            raise ValueError("native graph requires a bounded nonempty node tuple")
        for index, node in enumerate(self.nodes):
            if type(node) is not NativeCausalNode or any(p >= index for p in node.equation.parents):
                raise ValueError("native nodes require parent-before-child order")
            if node.equation.parents and node.equation.function.input_width != sum(
                self.nodes[p].state_width for p in node.equation.parents
            ):
                raise ValueError("native equation inputs must match explicit encoded parent widths")
        if type(self.observed) is not tuple or len(self.observed) != self.task.features:
            raise ValueError("native observed width must match task controls")
        for ref in (*self.observed, self.target):
            if (
                type(ref) is not tuple
                or len(ref) != 2
                or any(type(i) is not int for i in ref)
                or (
                    not 0 <= ref[0] < len(self.nodes)
                    or not 0 <= ref[1] < self.nodes[ref[0]].value_width
                )
            ):
                raise ValueError("native readout must reference a valid variable/channel")
        if self.target in self.observed:
            raise ValueError("native target variable cannot be observed")
        target = self.nodes[self.target[0]].domain
        expected = (
            "regression"
            if target.kind == "continuous"
            else "binary"
            if len(target.vocabulary) == 2
            else "multiclass"
        )
        if self.task.task_type != expected or self.task.classes != len(target.vocabulary):
            raise ValueError("native target domain must match task type and declared classes")
        self.admit()
        native_certificates(self.nodes)

    def admit(self) -> dict[str, int]:
        """Bound retained emissions, one-hot states and JSON copies before evaluating."""
        widths = tuple(n.state_width for n in self.nodes)
        if max(widths) > self.limits.node_channels or sum(widths) > self.limits.total_channels:
            raise ValueError("native encoded channel limit exceeded")
        report = admit_functional_graph(
            tuple(n.equation for n in self.nodes),
            self.limits,
            rows=self.task.rows,
            metadata=(self.task, self.nodes, self.observed, self.target, self.limits),
            features=self.task.features,
            outputs=self.nodes[self.target[0]].state_width,
        )
        # Retain score arrays, integer values, probabilities and raw uniform variates.
        # JSON float objects/text and temporary copies are charged conservatively too.
        channels = sum(
            n.state_width + n.value_width + n.equation.function.output_width + 1 for n in self.nodes
        )
        source_bound = sum(
            (n.equation.root.width if n.equation.root else 0) + n.equation.function.output_width + 1
            for n in self.nodes
        )
        extra = 256 * self.task.rows * channels + 256 * sum(widths) * source_bound
        report["estimated_working_bytes"] += extra
        if report["estimated_working_bytes"] > self.limits.working_bytes:
            raise ValueError("estimated native graph working memory exceeds declared limit")
        return report


def native_certificates(nodes: tuple[NativeCausalNode, ...]) -> None:
    """Validate population moment contracts using typed parent state certificates."""
    states: list[list[ChannelCertificate]] = []
    for index, node in enumerate(nodes):
        state, _ = functional_node_certificates(
            node.equation, index, [c for p in node.equation.parents for c in states[p]]
        )
        if node.domain.kind != "continuous":
            sources = frozenset[tuple[int, int]]().union(*(c.sources for c in state))
            if node.domain.emission == "softmax":
                sources |= {
                    (
                        index,
                        node.equation.function.output_width
                        + (node.equation.root.width if node.equation.root else 0),
                    )
                }
            state = [ChannelCertificate(sources)] * node.state_width
        states.append(state)
