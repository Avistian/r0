"""Independent topology and complete joint mechanism priors on random-size forests."""

from dataclasses import dataclass

from r0.modules.priors.lib.causal_limits import CausalLimits
from r0.modules.priors.lib.hyperprior_graph_admission import hyperprior_readouts
from r0.modules.priors.lib.joint_function_prior import JointFunctionPrior
from r0.modules.priors.lib.joint_graph_request import JointGraphNoise, JointGraphRequest
from r0.modules.priors.lib.joint_hyperprior_admission import (
    joint_hyperprior_budget,
    joint_hyperprior_child_limits,
    joint_hyperprior_child_raw,
)
from r0.modules.priors.lib.joint_root_plans import JointRootPrior
from r0.modules.priors.lib.linear import LinearRecipe
from r0.modules.priors.lib.node_pipeline import CausalNodePipeline
from r0.modules.priors.lib.raw_draw_records import RawDrawLimits
from r0.modules.priors.lib.topology_hyperprior import (
    FixedSizeTopologyHyperprior,
    TopologyHyperprior,
    TopologyHyperpriorRequest,
)


@dataclass(frozen=True, slots=True)
class JointHyperpriorGraphRequest:
    task: LinearRecipe
    topology: TopologyHyperpriorRequest
    width: int = 1
    calibration_rows: int = 256
    root_prior: JointRootPrior = JointRootPrior()
    function_prior: JointFunctionPrior = JointFunctionPrior()
    noise: JointGraphNoise = JointGraphNoise()
    pipeline: CausalNodePipeline | None = None
    mode: str = "meta"
    categorical_policy: str = "shared"
    global_policy: str = "scalar-quantile-v1"
    limits: CausalLimits = CausalLimits()
    raw_limits: RawDrawLimits = RawDrawLimits()
    layout: str = "prefix-features-suffix-targets-r0-v1"
    version: str = "joint-hyperprior-graph-request-v1"

    def __post_init__(self) -> None:
        if (
            type(self.task) is not LinearRecipe
            or type(self.topology) not in (TopologyHyperprior, FixedSizeTopologyHyperprior)
            or type(self.width) is not int
            or not 1 <= self.width < 2**63
            or type(self.calibration_rows) is not int
            or not 1 <= self.calibration_rows < 2**63
            or type(self.root_prior) is not JointRootPrior
            or type(self.function_prior) is not JointFunctionPrior
            or type(self.noise) is not JointGraphNoise
            or type(self.limits) is not CausalLimits
            or type(self.raw_limits) is not RawDrawLimits
            or self.version != "joint-hyperprior-graph-request-v1"
            or self.layout != "prefix-features-suffix-targets-r0-v1"
            or self.mode not in ("meta", "global", "local")
            or self.categorical_policy not in ("shared", "refresh")
            or self.global_policy not in ("scalar-quantile-v1", "strict-contract-v1")
        ):
            raise ValueError("joint hyperprior request requires complete versioned source controls")
        from r0.modules.priors.lib.joint_hyperprior_contracts import validate_hyperprior_catalogs

        validate_hyperprior_catalogs(
            self.root_prior,
            self.function_prior,
            has_edges=self.topology.maximum_nodes > self.topology.components,
            mode=self.mode,
            global_policy=self.global_policy,
        )
        self.noise.validate(self.width)
        if self.pipeline is not None and (
            type(self.pipeline) is not CausalNodePipeline or self.pipeline.channels != self.width
        ):
            raise ValueError("joint hyperprior pipeline width differs")
        hyperprior_readouts(self.task, self.topology.minimum_nodes, self.width)
        self.admit()

    def admit(self) -> dict[str, int]:
        return joint_hyperprior_budget(self)

    def for_parents(self, parents: tuple[tuple[int, ...], ...]) -> JointGraphRequest:
        n = len(parents)
        if not self.topology.minimum_nodes <= n <= self.topology.maximum_nodes:
            raise ValueError("joint hyperprior realized size differs from support")
        observed, target = hyperprior_readouts(self.task, n, self.width)
        return JointGraphRequest(
            self.task,
            parents,
            (self.width,) * n,
            observed,
            target,
            self.calibration_rows,
            self.root_prior,
            self.function_prior,
            (self.noise,) * n,
            (self.pipeline,) * n,
            self.mode,
            self.categorical_policy,
            self.global_policy,
            joint_hyperprior_child_limits(self),
            joint_hyperprior_child_raw(self),
        )
