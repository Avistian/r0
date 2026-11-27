"""Potential exogenous ancestry distinguishes converted readouts from later state noise."""

from r0.modules.priors.lib.causal_activations import CausalActivation
from r0.modules.priors.lib.causal_nodes import ActivatedCausalNode, CausalNode, StochasticCausalNode
from r0.modules.priors.lib.node_pipeline import CausalNodePipeline

type Sources = list[set[tuple[int, int]]]


def activation_sources(sources: Sources, activation: CausalActivation) -> Sources:
    if not activation.uses_input(len(sources)):
        return [set[tuple[int, int]]() for _ in sources]
    if activation.mixes_channels:
        shared = set[tuple[int, int]]().union(*sources)
        return [set(shared) for _ in sources]
    return [set(source) for source in sources]


def pipeline_ancestry(
    nodes: tuple[CausalNode, ...], pipelines: tuple[CausalNodePipeline, ...]
) -> dict[tuple[int, int], set[tuple[int, int]]]:
    states: list[Sources] = []
    readouts: dict[tuple[int, int], set[tuple[int, int]]] = {}
    for index, (node, pipeline) in enumerate(zip(nodes, pipelines, strict=True)):
        assert isinstance(node, ActivatedCausalNode)
        incoming = [source for parent in node.parents for source in states[parent]]
        incoming = activation_sources(incoming, node.input_activation)
        affine = [
            set[tuple[int, int]]().union(
                *(source for i, source in enumerate(incoming) if node.weights[i][channel] != 0)
            )
            for channel in range(len(node.bias))
        ]
        before = activation_sources(affine, node.output_activation)
        for stage in pipeline.stages:
            before = activation_sources(before, stage)
        if pipeline.channel_importance:
            before = [
                source if factor != 0 else set[tuple[int, int]]()
                for source, factor in zip(before, pipeline.channel_importance, strict=True)
            ]
        if isinstance(node, StochasticCausalNode):
            noise = [
                {(index, source) for source in sources}
                if node.has_noise_scale(channel)
                else set[tuple[int, int]]()
                for channel, sources in enumerate(node.exogenous.channel_sources(len(node.bias)))
            ]
            if node.noise_response.strength:
                noise = [
                    sources | affine[j] if sources else sources for j, sources in enumerate(noise)
                ]
        else:
            noise = [
                {(index, j)} if scale != 0 else set[tuple[int, int]]()
                for j, scale in enumerate(node.noise_scale)
            ]
        state = [a | n for a, n in zip(before, noise, strict=True)]
        captured = {c.channel for c in pipeline.converters}
        for channel, sources in enumerate(state):
            readouts[index, channel] = set(before[channel] if channel in captured else sources)
        states.append(state)
    return readouts
