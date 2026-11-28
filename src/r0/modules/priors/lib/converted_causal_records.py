"""Typed category captures alongside width-preserving structural state pipelines."""

from dataclasses import dataclass

from r0.modules.priors.lib.categorical_converter import CategoricalConverter
from r0.modules.priors.lib.causal_limits import CausalLimits
from r0.modules.priors.lib.frozen_function_evaluation import (
    admit_frozen_function,
    function_geometry,
)
from r0.modules.priors.lib.functional_analysis import (
    ChannelCertificate,
    functional_node_certificates,
)
from r0.modules.priors.lib.functional_causal_node import FunctionalCausalNode
from r0.modules.priors.lib.functional_graph_admission import (
    admit_functional_graph,
    validate_functional_nodes,
)
from r0.modules.priors.lib.linear import LinearRecipe
from r0.modules.priors.lib.training_features import TrainingFeature


@dataclass(frozen=True, slots=True)
class ConvertedReadout:
    node: int
    channel: int | None = None
    capture: str | None = None

    def __post_init__(self) -> None:
        if (
            type(self.node) is not int
            or self.node < 0
            or (self.channel is None) == (self.capture is None)
            or self.channel is not None
            and (type(self.channel) is not int or self.channel < 0)
            or self.capture is not None
            and (type(self.capture) is not str or not self.capture)
        ):
            raise ValueError(
                "readout must select a node and exactly one numeric channel or named capture"
            )


@dataclass(frozen=True, slots=True)
class ConvertedCausalNode:
    equation: FunctionalCausalNode
    converters: tuple[CategoricalConverter, ...] = ()
    version: str = "converted-causal-node-v1"

    def __post_init__(self) -> None:
        if (
            self.version != "converted-causal-node-v1"
            or type(self.equation) is not FunctionalCausalNode
            or type(self.converters) is not tuple
            or any(type(c) is not CategoricalConverter for c in self.converters)
            or len({c.name for c in self.converters}) != len(self.converters)
        ):
            raise ValueError(
                "converted node requires its frozen equation and unique named converters"
            )
        occupied: set[int] = (
            {c.channel for c in self.equation.pipeline.converters}
            if self.equation.pipeline
            else set()
        )
        for converter in self.converters:
            if converter.stop > self.equation.function.output_width or any(
                j in occupied for j in range(converter.start, converter.stop)
            ):
                raise ValueError(
                    "categorical converter spans must be valid and disjoint from every converter"
                )
            occupied.update(range(converter.start, converter.stop))

    def feature(self, ref: ConvertedReadout) -> TrainingFeature:
        if ref.capture is not None:
            converter = next((c for c in self.converters if c.name == ref.capture), None)
            if converter is None:
                raise ValueError("unknown categorical capture")
            return TrainingFeature(
                "binary" if len(converter.vocabulary) == 2 else "categorical", converter.vocabulary
            )
        assert ref.channel is not None
        if ref.channel >= self.equation.function.output_width or any(
            c.start <= ref.channel < c.stop for c in self.converters
        ):
            raise ValueError("numeric readout must select an available non-categorical channel")
        return TrainingFeature()


@dataclass(frozen=True, slots=True)
class ConvertedCausalRecipe:
    task: LinearRecipe
    nodes: tuple[ConvertedCausalNode, ...]
    observed: tuple[ConvertedReadout, ...]
    target: ConvertedReadout
    limits: CausalLimits = CausalLimits()
    version: str = "converted-causal-v1"

    def __post_init__(self) -> None:
        if (
            self.version != "converted-causal-v1"
            or type(self.task) is not LinearRecipe
            or type(self.limits) is not CausalLimits
            or type(self.nodes) is not tuple
            or not self.nodes
            or any(type(n) is not ConvertedCausalNode for n in self.nodes)
        ):
            raise ValueError("converted graph requires its frozen task, nodes and limits")
        validate_functional_nodes(tuple(n.equation for n in self.nodes), self.limits)
        if self.task.noise_ratio or self.task.separation != 1 or self.task.class_offsets:
            raise ValueError("direct captured targets require default adapter-only controls")
        if type(self.observed) is not tuple or len(self.observed) != self.task.features:
            raise ValueError("converted observed references must match feature count")
        for ref in (*self.observed, self.target):
            if type(ref) is not ConvertedReadout or ref.node >= len(self.nodes):
                raise ValueError("converted readout must select a valid node")
            self.nodes[ref.node].feature(ref)
        if self.target in self.observed:
            raise ValueError("target capture cannot be observed")
        feature = self.nodes[self.target.node].feature(self.target)
        expected = (
            "regression"
            if feature.kind == "continuous"
            else "binary"
            if feature.kind == "binary"
            else "multiclass"
        )
        if self.task.task_type != expected or self.task.classes != len(feature.vocabulary):
            raise ValueError("captured target type/classes differ from task controls")
        self.admit()
        converted_certificates(self.nodes)

    def admit(self) -> dict[str, int]:
        return admit_converted_nodes(
            self.nodes,
            self.task.rows,
            self.limits,
            metadata=(self.task, self.nodes, self.observed, self.target, self.limits),
            features=self.task.features,
            outputs=max(1, len(self.nodes[self.target.node].feature(self.target).vocabulary)),
        )


def admit_converted_nodes(
    nodes: tuple[ConvertedCausalNode, ...],
    rows: int,
    limits: CausalLimits,
    *,
    metadata: tuple[object, ...],
    features: int = 0,
    outputs: int = 0,
) -> dict[str, int]:
    """Charge post functions as nested mechanisms, including center-dependent row floors.

    The common graph admission already counts all frozen parameters in metadata.
    Summed post-function reports conservatively include their standalone buffers;
    their own geometry also passes the caller's channel limits before any tensors.
    """
    report = admit_functional_graph(
        tuple(n.equation for n in nodes),
        limits,
        rows=rows,
        metadata=metadata,
        features=features,
        outputs=outputs,
    )
    work = sum(
        8 * rows * len(c.vocabulary) * (3 * c.width + 8) + 16 * len(c.vocabulary) * c.width
        for n in nodes
        for c in n.converters
    )
    for node in nodes:
        for converter in node.converters:
            if converter.post_function is None:
                continue
            nested = admit_frozen_function(converter.post_function, rows, limits)
            dimensions, _, _ = function_geometry(converter.post_function)
            report["largest_node_channels"] = max(report["largest_node_channels"], max(dimensions))
            work += nested["estimated_working_bytes"]
    report["estimated_working_bytes"] += work
    if report["estimated_working_bytes"] > limits.working_bytes:
        raise ValueError(
            "categorical converter and nested post-function workspace "
            "exceeds limit before allocation"
        )
    return report


def converted_certificates(
    nodes: tuple[ConvertedCausalNode, ...],
) -> list[list[ChannelCertificate]]:
    states: list[list[ChannelCertificate]] = []
    for index, node in enumerate(nodes):

        def convert(
            before: list[ChannelCertificate], node: ConvertedCausalNode = node, index: int = index
        ) -> list[ChannelCertificate]:
            result = list(before)
            for j, converter in enumerate(node.converters):
                part = before[converter.start : converter.stop]
                if converter.stochastic and any(c.order <= 2 for c in part):
                    raise ValueError(
                        "softmax standardization requires certified finite second moments"
                    )
                sources = frozenset[tuple[int, int]]().union(*(c.sources for c in part))
                if converter.stochastic:
                    sources |= {
                        (
                            index,
                            node.equation.function.output_width
                            + (node.equation.root.width if node.equation.root else 0)
                            + j,
                        )
                    }
                if not converter.mode.endswith("_id"):
                    result[converter.start : converter.stop] = [
                        ChannelCertificate(sources)
                    ] * converter.width
            return result

        state, _ = functional_node_certificates(
            node.equation,
            index,
            [c for p in node.equation.parents for c in states[p]],
            convert=convert,
        )
        states.append(state)
    return states
