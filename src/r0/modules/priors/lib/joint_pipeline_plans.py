"""Frozen source numerical node-pipeline decisions in the shared named scope."""

from dataclasses import dataclass

from r0.modules.priors.lib.activation_sampling import finite_number
from r0.modules.priors.lib.joint_converter_plans import (
    JointConverterPlan,
    JointNumericalConverterPrior,
    sample_joint_converter,
)
from r0.modules.priors.lib.joint_innovation_plans import (
    JointInnovationPlan,
    JointInnovationPrior,
    sample_joint_innovation,
)
from r0.modules.priors.lib.joint_plan_bindings import numerical_binding, operation_events
from r0.modules.priors.lib.joint_plan_ownership import JointPlanOwner, plan_owners
from r0.modules.priors.lib.joint_sampling import JointSamplingContext
from r0.modules.priors.lib.joint_sampling_records import JointSamplingRecord, validate_path
from r0.modules.priors.lib.joint_weight_plans import JointWeightPlan, sample_joint_weights
from r0.modules.priors.lib.parameter_preferences import NumericalParameter


@dataclass(frozen=True, slots=True)
class JointPipelinePrior:
    channel_importance: bool = True
    l2_normalization: bool = True
    node_importance: bool = True
    gaussian_noise: bool = False
    weight_family: str = "polynomial"
    scalarization: str = "scalarized-weight-batches-v1"
    converter: JointNumericalConverterPrior = JointNumericalConverterPrior()
    version: str = "independent-node-pipeline-prior-v1"

    def __post_init__(self) -> None:
        if (
            self.version
            != (
                "r0-innovation-pipeline-prior-v2"
                if type(self) is JointInnovationPipelinePrior
                else "independent-node-pipeline-prior-v1"
            )
            or any(
                type(v) is not bool
                for v in (
                    self.channel_importance,
                    self.l2_normalization,
                    self.node_importance,
                    self.gaussian_noise,
                )
            )
            or self.weight_family not in ("polynomial", "uniform")
            or self.scalarization
            not in ("scalarized-weight-batches-v1", "source-weight-batches-v1")
            or type(self.converter) is not JointNumericalConverterPrior
        ):
            raise ValueError("pipeline prior requires supported frozen controls")


@dataclass(frozen=True, slots=True)
class JointInnovationPipelinePrior(JointPipelinePrior):
    innovation: JointInnovationPrior = JointInnovationPrior()
    version: str = "r0-innovation-pipeline-prior-v2"

    def __post_init__(self) -> None:
        JointPipelinePrior.__post_init__(self)
        if type(self.innovation) is not JointInnovationPrior or self.gaussian_noise:
            raise ValueError("v2 innovation pipeline replaces the legacy Gaussian sigma switch")


def pipeline_scalar_request(role: str) -> NumericalParameter:
    if role == "sigma":
        return NumericalParameter("gaussian_noise_sigma", 1e-5, 1.0, scale="log")
    if role == "importance":
        return NumericalParameter("node_importance", 0.1, 10.0, scale="log")
    raise ValueError("unknown pipeline scalar role")


@dataclass(frozen=True, slots=True)
class JointPipelinePlan:
    operation: tuple[str, ...]
    width: int
    prior: JointPipelinePrior
    weights: JointWeightPlan | None
    converters: tuple[JointConverterPlan, ...]
    sigma: float
    importance: float
    version: str = "independent-node-pipeline-plan-v1"

    def __post_init__(self) -> None:
        validate_path(self.operation)
        if (
            self.version
            != (
                "r0-innovation-pipeline-plan-v2"
                if type(self) is JointInnovationPipelinePlan
                else "independent-node-pipeline-plan-v1"
            )
            or type(self.width) is not int
            or not 1 <= self.width < 2**63
            or type(self.prior) not in (JointPipelinePrior, JointInnovationPipelinePrior)
            or (self.weights is not None and type(self.weights) is not JointWeightPlan)
            or type(self.converters) is not tuple
            or any(type(c) is not JointConverterPlan for c in self.converters)
            or any(not finite_number(v) for v in (self.sigma, self.importance))
            or not 0 <= self.sigma <= 1
            or not 0.1 <= self.importance <= 10
        ):
            raise ValueError("pipeline plan requires frozen source geometry and values")
        if self.prior.channel_importance != (self.weights is not None):
            raise ValueError("pipeline weights differ from enabled importance")
        if self.weights is not None and (
            self.weights.operation != (*self.operation, "weights")
            or self.weights.width != self.width
            or self.weights.batch != 1
            or self.weights.family != self.prior.weight_family
            or self.weights.scalarization != self.prior.scalarization
        ):
            raise ValueError("pipeline weight binding differs")
        if tuple(c.channel for c in self.converters) != tuple(range(len(self.converters))):
            raise ValueError("pipeline converters must occupy ordered reserved channels")
        for i, converter in enumerate(self.converters):
            if (
                converter.channel >= self.width
                or converter.prior != self.prior.converter
                or converter.operation != (*self.operation, f"converter:{i}")
            ):
                raise ValueError("pipeline converter binding differs")
        if (not self.prior.gaussian_noise and self.sigma != 0) or (
            not self.prior.node_importance and self.importance != 1
        ):
            raise ValueError("disabled pipeline scalars must have identity values")

    def owners(self) -> tuple[JointPlanOwner, ...]:
        result = [JointPlanOwner(self.operation, "pipeline", ())]
        if self.weights is not None:
            result.extend(plan_owners(self.weights, ("weights",)))
        for i, converter in enumerate(self.converters):
            result.extend(plan_owners(converter, ("converters", i)))
        if isinstance(self, JointInnovationPipelinePlan):
            assert self.innovation is not None
            result.append(JointPlanOwner(self.innovation.operation, "innovation", ("innovation",)))
        return tuple(result)

    def validate(self, decisions: JointSamplingRecord) -> None:
        if isinstance(self, JointInnovationPipelinePlan):
            assert self.innovation is not None
            self.innovation.validate(decisions)
        own = tuple(
            e for e in operation_events(decisions, self.operation) if e.operation == self.operation
        )
        roles = (("sigma",) if self.prior.gaussian_noise else ()) + (
            ("importance",) if self.prior.node_importance else ()
        )
        if tuple(e.role for e in own) != roles:
            raise ValueError("pipeline scalar decision ownership differs")
        for event in own:
            expected = numerical_binding(event, decisions, pipeline_scalar_request(event.role))
            actual = self.sigma if event.role == "sigma" else self.importance
            if expected.values != (actual,):
                raise ValueError("pipeline scalar differs from retained decision")
        if self.weights is not None:
            self.weights.validate(decisions)
        for converter in self.converters:
            converter.validate(decisions)
        paths = {owner.operation for owner in self.owners()}
        if any(e.operation not in paths for e in operation_events(decisions, self.operation)):
            raise ValueError("pipeline has foreign decision owners")


@dataclass(frozen=True, slots=True)
class JointInnovationPipelinePlan(JointPipelinePlan):
    innovation: JointInnovationPlan | None = None
    version: str = "r0-innovation-pipeline-plan-v2"

    def __post_init__(self) -> None:
        JointPipelinePlan.__post_init__(self)
        if (
            type(self.prior) is not JointInnovationPipelinePrior
            or type(self.innovation) is not JointInnovationPlan
            or self.innovation.width != self.width
            or self.innovation.prior != self.prior.innovation
            or self.innovation.operation != (*self.operation, "innovation")
        ):
            raise ValueError("pipeline innovation binding differs")


def sample_joint_pipeline(
    context: JointSamplingContext,
    name: str,
    *,
    width: int,
    groups: tuple[str, ...],
    prior: JointPipelinePrior,
) -> JointPipelinePlan:
    with context.transaction(name) as operation:
        weights = (
            sample_joint_weights(
                operation,
                "weights",
                width=width,
                family=prior.weight_family,
                scalarization=prior.scalarization,
            )
            if prior.channel_importance
            else None
        )
        converters = tuple(
            sample_joint_converter(
                operation,
                f"converter:{i}",
                channel=i,
                feature_group=group,
                prior=prior.converter,
            )
            for i, group in enumerate(groups)
        )
        sigma = (
            operation.numerical("sigma", pipeline_scalar_request("sigma")).values[0]
            if prior.gaussian_noise
            else 0.0
        )
        importance = (
            operation.numerical("importance", pipeline_scalar_request("importance")).values[0]
            if prior.node_importance
            else 1.0
        )
        args = (operation.operation, width, prior, weights, converters, sigma, importance)
        result = (
            JointInnovationPipelinePlan(
                *args,
                innovation=sample_joint_innovation(operation, width=width, prior=prior.innovation),
            )
            if isinstance(prior, JointInnovationPipelinePrior)
            else JointPipelinePlan(*args)
        )
        result.validate(operation.freeze())
        return result
