"""Semantic plan owners derived from concrete plan trees, including empty scopes."""

from dataclasses import dataclass

from r0.modules.priors.lib.joint_activation_plans import JointActivationPlan
from r0.modules.priors.lib.joint_converter_plans import JointConverterPlan
from r0.modules.priors.lib.joint_function_plans import JointFunctionPlan
from r0.modules.priors.lib.joint_matrix_plans import JointMatrixPlan
from r0.modules.priors.lib.joint_point_plans import JointPointPlan
from r0.modules.priors.lib.joint_root_plans import JointRootPlan
from r0.modules.priors.lib.joint_weight_plans import JointWeightPlan

type JointOwnedPlan = (
    JointRootPlan
    | JointFunctionPlan
    | JointMatrixPlan
    | JointPointPlan
    | JointWeightPlan
    | JointActivationPlan
    | JointConverterPlan
)
type PlanReference = tuple[str | int, ...]


@dataclass(frozen=True, slots=True)
class JointPlanOwner:
    """A typed owner and a field path resolving to its retained graph-plan record."""

    operation: tuple[str, ...]
    kind: str
    reference: PlanReference


def plan_owners(plan: JointOwnedPlan, reference: PlanReference) -> tuple[JointPlanOwner, ...]:
    children: list[tuple[JointOwnedPlan, PlanReference]] = []
    if type(plan) is JointRootPlan:
        kind = "root"
        if plan.weights is not None:
            children.append((plan.weights, (*reference, "weights")))
        children.append((plan.function, (*reference, "function")))
    elif type(plan) is JointFunctionPlan:
        kind = "function"
        if plan.family == "mlp":
            for child in plan.ordered_children:
                field = "matrices" if type(child) is JointMatrixPlan else "activations"
                collection = plan.matrices if field == "matrices" else plan.activations
                index = next(i for i, item in enumerate(collection) if item is child)
                children.append((child, (*reference, field, index)))
        else:
            if plan.weights is not None:
                children.append((plan.weights, (*reference, "weights")))
            children.extend(
                (item, (*reference, "matrices", i)) for i, item in enumerate(plan.matrices)
            )
            children.extend(
                (item, (*reference, "children", i)) for i, item in enumerate(plan.children)
            )
    elif type(plan) is JointMatrixPlan:
        kind = "matrix"
        for field, value in (
            ("weights", plan.weights),
            ("points", plan.points),
            ("child", plan.child),
            ("activation", plan.activation),
        ):
            if value is not None:
                children.append((value, (*reference, field)))
    elif type(plan) is JointPointPlan:
        kind = "points"
        children.extend(
            (
                (plan.mixture_weights, (*reference, "mixture_weights")),
                (plan.factor_weights, (*reference, "factor_weights")),
            )
        )
    elif type(plan) is JointWeightPlan:
        kind = "weights"
    elif type(plan) is JointActivationPlan:
        kind = "activation"
    elif type(plan) is JointConverterPlan:
        kind = "converter"
        if plan.activation is not None:
            children.append((plan.activation, (*reference, "activation")))
    else:
        raise ValueError("typed ownership requires a supported concrete plan")
    result = [JointPlanOwner(plan.operation, kind, reference)]
    for child, child_reference in children:
        result.extend(plan_owners(child, child_reference))
    return tuple(result)
