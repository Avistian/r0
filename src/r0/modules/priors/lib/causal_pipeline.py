"""Authored causal pipelines with separate state and readout channels."""

from dataclasses import dataclass

from r0.modules.priors.lib.causal import CausalRecipe
from r0.modules.priors.lib.causal_moments import causal_moment_orders
from r0.modules.priors.lib.causal_nodes import ActivatedCausalNode
from r0.modules.priors.lib.node_pipeline import CausalNodePipeline


@dataclass(frozen=True, slots=True)
class CausalPipelineRecipe(CausalRecipe):
    pipelines: tuple[CausalNodePipeline, ...] = ()

    @property
    def node_pipelines(self) -> tuple[CausalNodePipeline, ...] | None:
        return self.pipelines

    @property
    def sampling_parameter_count(self) -> int:
        return sum(p.parameter_count for p in self.pipelines)

    def __post_init__(self) -> None:
        if (
            self.version != "causal-v2"
            or self.construction != "authored"
            or type(self.pipelines) is not tuple
            or len(self.pipelines) != len(self.nodes)
            or any(type(p) is not CausalNodePipeline for p in self.pipelines)
        ):
            raise ValueError("authored pipelines require one frozen pipeline per causal-v2 node")
        for node, pipeline in zip(self.nodes, self.pipelines, strict=True):
            if (
                not isinstance(node, ActivatedCausalNode)
                or node.noise_position != "after_output"
                or len(node.bias) != pipeline.channels
            ):
                raise ValueError("pipeline nodes require matching width and after-output noise")
        CausalRecipe.__post_init__(self)
        causal_moment_orders(self.nodes, pipelines=self.pipelines)
