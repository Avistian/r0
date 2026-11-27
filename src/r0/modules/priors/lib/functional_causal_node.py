"""Frozen structural node records shared by complete tasks and ancestral prefixes."""

from dataclasses import dataclass

from r0.modules.priors.lib.activation_sampling import finite_number
from r0.modules.priors.lib.exogenous import ExogenousDistribution, NoiseResponse
from r0.modules.priors.lib.frozen_function_evaluation import FrozenFunction, function_geometry
from r0.modules.priors.lib.joint_root_base import JointRootBase
from r0.modules.priors.lib.node_pipeline import CausalNodePipeline


@dataclass(frozen=True, slots=True)
class FunctionalCausalNode:
    parents: tuple[int, ...]
    function: FrozenFunction
    root: JointRootBase | None = None
    noise_scale: tuple[float, ...] = ()
    exogenous: ExogenousDistribution = ExogenousDistribution()
    noise_response: NoiseResponse = NoiseResponse()
    pipeline: CausalNodePipeline | None = None
    version: str = "functional-causal-node-v1"

    def __post_init__(self) -> None:
        if self.version != "functional-causal-node-v1":
            raise ValueError("unsupported functional node version")
        function_geometry(self.function)
        width = self.function.output_width
        if (
            type(self.noise_scale) is not tuple
            or self.noise_scale
            and (
                len(self.noise_scale) != width
                or any(not finite_number(v) or v < 0 for v in self.noise_scale)
            )
        ):
            raise ValueError("noise scales must be nonnegative finite output-channel factors")
        if (
            type(self.exogenous) is not ExogenousDistribution
            or type(self.noise_response) is not NoiseResponse
        ):
            raise ValueError("functional noise requires frozen exogenous and response descriptors")
        self.exogenous.validate_width(width)
        if self.pipeline is not None and (
            type(self.pipeline) is not CausalNodePipeline or self.pipeline.channels != width
        ):
            raise ValueError("functional pipeline width must match function output")
        if (
            type(self.parents) is not tuple
            or any(type(p) is not int or p < 0 for p in self.parents)
            or len(set(self.parents)) != len(self.parents)
        ):
            raise ValueError("functional parents must be distinct nonnegative integer indices")
        if self.parents:
            if self.root is not None:
                raise ValueError("only parentless functions have a root base")
        elif type(self.root) is not JointRootBase or self.root.width != self.function.input_width:
            raise ValueError("root base width must match the root function input")
