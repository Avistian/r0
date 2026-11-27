"""Explicit graph inputs, source priors, structural innovations and resource policy."""

from dataclasses import dataclass

from r0.modules.priors.lib.activation_sampling import finite_number
from r0.modules.priors.lib.causal_limits import CausalLimits
from r0.modules.priors.lib.exogenous import ExogenousDistribution, NoiseResponse
from r0.modules.priors.lib.joint_function_prior import JointFunctionPrior
from r0.modules.priors.lib.joint_graph_plans import validate_graph_geometry
from r0.modules.priors.lib.joint_root_plans import JointRootPrior
from r0.modules.priors.lib.linear import LinearRecipe
from r0.modules.priors.lib.node_pipeline import CausalNodePipeline
from r0.modules.priors.lib.raw_draw_records import RawDrawLimits


@dataclass(frozen=True, slots=True)
class JointGraphNoise:
    scales: tuple[float, ...] = ()
    exogenous: ExogenousDistribution = ExogenousDistribution()
    response: NoiseResponse = NoiseResponse()

    def validate(self, width: int) -> None:
        if (
            type(self.scales) is not tuple
            or (self.scales and len(self.scales) != width)
            or any(not finite_number(v) or v < 0 for v in self.scales)
        ):
            raise ValueError("graph structural noise must have finite nonnegative channel scales")
        if (
            type(self.exogenous) is not ExogenousDistribution
            or type(self.response) is not NoiseResponse
        ):
            raise ValueError("graph structural noise requires frozen laws")
        self.exogenous.validate_width(width)


@dataclass(frozen=True, slots=True)
class JointGraphRequest:
    task: LinearRecipe
    parents: tuple[tuple[int, ...], ...]
    widths: tuple[int, ...]
    observed: tuple[tuple[int, int], ...]
    target: tuple[tuple[int, int], ...]
    calibration_rows: int = 256
    root_prior: JointRootPrior = JointRootPrior()
    function_prior: JointFunctionPrior = JointFunctionPrior()
    noise: tuple[JointGraphNoise, ...] = ()
    authored_pipelines: tuple[CausalNodePipeline | None, ...] = ()
    mode: str = "meta"
    categorical_policy: str = "shared"
    global_policy: str = "scalar-quantile-v1"
    limits: CausalLimits = CausalLimits()
    raw_limits: RawDrawLimits = RawDrawLimits()
    version: str = "joint-graph-request-v1"

    def __post_init__(self) -> None:
        if (
            self.version != "joint-graph-request-v1"
            or type(self.task) is not LinearRecipe
            or type(self.root_prior) is not JointRootPrior
            or type(self.function_prior) is not JointFunctionPrior
            or type(self.raw_limits) is not RawDrawLimits
        ):
            raise ValueError("graph request requires its version and frozen task/priors/limits")
        validate_graph_geometry(self.parents, self.widths, self.calibration_rows, self.limits)
        for refs in (self.observed, self.target):
            if type(refs) is not tuple or any(
                type(ref) is not tuple
                or len(ref) != 2
                or any(type(i) is not int for i in ref)
                or not 0 <= ref[0] < len(self.widths)
                or not 0 <= ref[1] < self.widths[ref[0]]
                for ref in refs
            ):
                raise ValueError("graph readouts must name declared node channels")
        outputs = self.task.classes if self.task.task_type == "multiclass" else 1
        if (
            len(self.observed) != self.task.features
            or len(self.target) != outputs
            or set(self.observed) & set(self.target)
        ):
            raise ValueError("graph readouts differ from task geometry or expose targets")
        if type(self.noise) is not tuple or self.noise and len(self.noise) != len(self.widths):
            raise ValueError("graph noise must align with nodes")
        for width, noise in zip(self.widths, self.noise, strict=False):
            if type(noise) is not JointGraphNoise:
                raise ValueError("graph noise must be frozen")
            noise.validate(width)
        if (
            type(self.authored_pipelines) is not tuple
            or self.authored_pipelines
            and len(self.authored_pipelines) != len(self.widths)
        ):
            raise ValueError("authored pipelines must align with graph nodes")
        for width, pipeline in zip(self.widths, self.authored_pipelines, strict=False):
            if pipeline is not None and (
                type(pipeline) is not CausalNodePipeline or pipeline.channels != width
            ):
                raise ValueError("authored pipeline geometry differs from graph node")
