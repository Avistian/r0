"""Versioned random-size topology composed with four existing mechanism families."""

from dataclasses import dataclass

from r0.modules.priors.lib.causal_limits import CausalLimits
from r0.modules.priors.lib.fixed_parent_topology import FixedParentTopology
from r0.modules.priors.lib.joint_function_prior import JointFunctionPrior
from r0.modules.priors.lib.joint_pipeline_plans import (
    JointInnovationPipelinePrior,
    JointPipelinePrior,
)
from r0.modules.priors.lib.joint_root_plans import JointRootPrior
from r0.modules.priors.lib.joint_width_plans import JointWidthPrior
from r0.modules.priors.lib.linear import LinearRecipe
from r0.modules.priors.lib.parent_composition_plans import ParentCompositionPrior
from r0.modules.priors.lib.raw_draw_records import RawDrawLimits
from r0.modules.priors.lib.structural_topology_request import StructuralTopologyRequest
from r0.modules.priors.lib.topology_hyperprior import (
    FixedSizeTopologyHyperprior,
    TopologyHyperprior,
    TopologyHyperpriorRequest,
)
from r0.modules.priors.lib.typed_topology_roles import ConvertedTopologyRoles, NativeTopologyRoles


@dataclass(frozen=True, slots=True)
class TypedTopologyRequest:
    topology: TopologyHyperpriorRequest | FixedParentTopology | StructuralTopologyRequest
    task: LinearRecipe
    family: str = "functional"
    width_prior: JointWidthPrior = JointWidthPrior()
    native: NativeTopologyRoles | None = None
    converted: ConvertedTopologyRoles | None = None
    pipeline: JointPipelinePrior | None = None
    subsample: bool = True
    calibration_rows: int = 256
    root_prior: JointRootPrior = JointRootPrior()
    function_prior: JointFunctionPrior = JointFunctionPrior()
    mode: str = "meta"
    categorical_policy: str = "shared"
    global_policy: str = "scalar-quantile-v1"
    limits: CausalLimits = CausalLimits()
    raw_limits: RawDrawLimits = RawDrawLimits()
    version: str = "typed-topology-request-r0-v1"

    def __post_init__(self) -> None:
        composed = type(self) is TypedParentTopologyRequest
        expected_version = (
            "typed-parent-topology-request-r0-v1" if composed else "typed-topology-request-r0-v1"
        )
        topology_types = (
            (
                TopologyHyperprior,
                FixedSizeTopologyHyperprior,
                FixedParentTopology,
                StructuralTopologyRequest,
            )
            if composed
            else (TopologyHyperprior, FixedSizeTopologyHyperprior)
        )
        if (
            self.version != expected_version
            or type(self.topology) not in topology_types
            or type(self.task) is not LinearRecipe
            or self.family not in ("functional", "native", "native-vector", "converted")
            or type(self.width_prior) is not JointWidthPrior
            or type(self.root_prior) is not JointRootPrior
            or type(self.function_prior) is not JointFunctionPrior
            or type(self.limits) is not CausalLimits
            or type(self.raw_limits) is not RawDrawLimits
            or type(self.subsample) is not bool
            or type(self.calibration_rows) is not int
            or not 1 <= self.calibration_rows < 2**63
            or self.mode not in ("meta", "global", "local")
            or self.categorical_policy not in ("shared", "refresh")
            or self.global_policy not in ("scalar-quantile-v1", "strict-contract-v1")
            or self.pipeline is not None
            and type(self.pipeline) not in (JointPipelinePrior, JointInnovationPipelinePrior)
        ):
            raise ValueError("typed topology requires complete versioned family controls")
        native = self.family in ("native", "native-vector")
        if (native and type(self.native) is not NativeTopologyRoles) or (
            not native and self.native is not None
        ):
            raise ValueError("native topology families require their explicit role laws")
        if (self.family == "converted" and type(self.converted) is not ConvertedTopologyRoles) or (
            self.family != "converted" and self.converted is not None
        ):
            raise ValueError("converted topology requires its explicit vocabulary laws")
        if self.family != "functional" and (
            self.task.active_features != self.task.features
            or self.task.signal_features != self.task.features
        ):
            raise ValueError("typed roles derive all feature metadata from their readouts")
        if native:
            if self.pipeline is not None or not self.subsample:
                raise ValueError(
                    "native role placement does not consume a numerical pipeline/subsample override"
                )
            if self.topology.minimum_nodes < self.task.features + 1:
                raise ValueError(
                    "native role placement needs distinct feature and target nodes at minimum size"
                )
            assert self.native is not None
            target = self.native.target
            expected = (
                "regression"
                if target.kinds == ("continuous",)
                else "binary"
                if target.kinds == ("binary",)
                else "multiclass"
            )
            if (
                self.task.task_type != expected
                or expected == "multiclass"
                and (
                    target.kinds != ("categorical",) or target.cardinalities != (self.task.classes,)
                )
            ):
                raise ValueError(
                    "native target role must guarantee the declared task type/cardinality"
                )
            # Native output controls are derived from the variable, not a latent-score adapter.
            if self.task.noise_ratio or self.task.class_offsets or self.task.separation != 1:
                raise ValueError("native targets do not consume latent-score adapter controls")
        if self.converted is not None:
            k = len(self.converted.target_vocabulary)
            expected = "regression" if k == 0 else "binary" if k == 2 else "multiclass"
            if (
                len(self.converted.vocabularies) != self.task.features
                or self.task.task_type != expected
                or k
                and self.task.classes != k
            ):
                raise ValueError("converted role vocabularies differ from task semantics")
            if self.pipeline is None:
                raise ValueError("converted topology requires an explicit numerical pipeline")
            if self.task.noise_ratio or self.task.class_offsets or self.task.separation != 1:
                raise ValueError("converted targets do not consume latent-score adapter controls")
        self.admit()

    def admit(self) -> dict[str, int]:
        from r0.modules.priors.lib.typed_topology_admission import admit_typed_topology

        return admit_typed_topology(self)


@dataclass(frozen=True, slots=True)
class TypedParentTopologyRequest(TypedTopologyRequest):
    composition: ParentCompositionPrior = ParentCompositionPrior()
    version: str = "typed-parent-topology-request-r0-v1"

    def admit(self) -> dict[str, int]:
        from r0.modules.priors.lib.typed_parent_admission import admit_typed_parent_topology

        return admit_typed_parent_topology(self)
