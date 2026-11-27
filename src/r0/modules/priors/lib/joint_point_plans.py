"""Named Gaussian-mixture decisions for a shared point cloud."""

from dataclasses import dataclass
from typing import Any

from r0.modules.priors.lib.joint_plan_bindings import numerical_binding, operation_events
from r0.modules.priors.lib.joint_sampling import JointSamplingContext
from r0.modules.priors.lib.joint_sampling_records import JointSamplingRecord, validate_path
from r0.modules.priors.lib.joint_weight_plans import (
    JointWeightPlan,
    joint_weight_plan_from_dict,
    sample_joint_weights,
)
from r0.modules.priors.lib.parameter_preferences import NumericalParameter


def point_count_request() -> NumericalParameter:
    return NumericalParameter("random_gmp_n_centers", 1, 16, scale="log", integer="floor")


@dataclass(frozen=True, slots=True)
class JointPointPlan:
    operation: tuple[str, ...]
    count: int
    dimension: int
    components: int
    mixture_weights: JointWeightPlan
    factor_weights: JointWeightPlan
    law: str = "gaussian-mixture"
    version: str = "joint-point-plan-v1"

    def __post_init__(self) -> None:
        validate_path(self.operation)
        if any(type(n) is not int or not 1 <= n < 2**63 for n in (self.count, self.dimension)):
            raise ValueError("point dimensions must be positive integers below 2**63")
        if self.version != "joint-point-plan-v1" or self.law != "gaussian-mixture":
            raise ValueError("unsupported joint point plan version or law")
        if type(self.components) is not int or not 1 <= self.components <= 15:
            raise ValueError("mixture component count must lie in [1,15]")
        for plan, name, batch, width in (
            (self.mixture_weights, "mixture-weights", 1, self.components),
            (self.factor_weights, "factor-weights", self.components, self.dimension),
        ):
            if type(plan) is not JointWeightPlan or (
                plan.operation != (*self.operation, name)
                or plan.batch != batch
                or plan.width != width
            ):
                raise ValueError("point weights must match their owned geometry")
        if (self.mixture_weights.family, self.mixture_weights.scalarization) != (
            self.factor_weights.family,
            self.factor_weights.scalarization,
        ):
            raise ValueError("point weights must share their configured family and policy")

    @property
    def parameter_count(self) -> int:
        return 3 + self.mixture_weights.parameter_count + self.factor_weights.parameter_count

    @property
    def realization_parameter_count(self) -> int:
        c, d, n = self.components, self.dimension, self.count
        return (
            c * d
            + c * d * d
            + self.mixture_weights.realization_parameter_count
            + self.factor_weights.realization_parameter_count
            + n
            + n * d
            + 2
            + c
        )

    def validate(self, decisions: JointSamplingRecord) -> None:
        events = operation_events(decisions, self.operation)
        if not events or events[0].operation != self.operation or events[0].role != "components":
            raise ValueError("mixture point plan requires its first component-count decision")
        draw = numerical_binding(events[0], decisions, point_count_request())
        if draw.values != (self.components,):
            raise ValueError("mixture component count differs from its named realization")
        self.mixture_weights.validate(decisions)
        self.factor_weights.validate(decisions)
        expected = (events[0],) + operation_events(decisions, self.mixture_weights.operation)
        expected += operation_events(decisions, self.factor_weights.operation)
        if events != expected:
            raise ValueError("point plan must own exactly its ordered descendant decisions")


def sample_joint_points(
    context: JointSamplingContext,
    name: str,
    *,
    count: int,
    dimension: int,
    vector_family: str = "polynomial",
    scalarization: str = "scalarized-weight-batches-v1",
) -> JointPointPlan:
    if type(context) is not JointSamplingContext:
        raise ValueError("point sampling requires a shared context")
    if any(type(n) is not int or not 1 <= n < 2**63 for n in (count, dimension)):
        raise ValueError("point dimensions must be positive integers below 2**63")
    with context.transaction(name) as operation:
        components = int(operation.numerical("components", point_count_request()).values[0])
        mixture = sample_joint_weights(
            operation,
            "mixture-weights",
            width=components,
            family=vector_family,
            scalarization=scalarization,
        )
        factors = sample_joint_weights(
            operation,
            "factor-weights",
            width=dimension,
            batch=components,
            family=vector_family,
            scalarization=scalarization,
        )
        plan = JointPointPlan(operation.operation, count, dimension, components, mixture, factors)
        plan.validate(operation.freeze())
    return plan


def joint_point_plan_from_dict(
    data: dict[str, Any], *, decisions: JointSamplingRecord
) -> JointPointPlan:
    fields = dict(data)
    path = fields["operation"]
    if type(path) not in (list, tuple) or not 1 <= len(path) <= 16:
        raise ValueError("point operation path must be a bounded array")
    fields["operation"] = tuple(path)
    for name in ("mixture_weights", "factor_weights"):
        fields[name] = joint_weight_plan_from_dict(fields[name], decisions=decisions)
    plan = JointPointPlan(**fields)
    plan.validate(decisions)
    return plan
