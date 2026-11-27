"""Validate native plan roles against their exact shared named decisions."""

from r0.modules.priors.lib.categorical_preferences import CategoricalParameter
from r0.modules.priors.lib.joint_plan_bindings import (
    categorical_binding,
    numerical_binding,
    operation_events,
)
from r0.modules.priors.lib.joint_sampling_records import JointSamplingRecord, validate_path
from r0.modules.priors.lib.native_domain_prior import (
    AsymmetricNativeDomainPlan,
    AsymmetricNativeDomainPrior,
    NativeDomainPlan,
)
from r0.modules.priors.lib.parameter_preferences import NumericalParameter


def validate_native_domain(plan: NativeDomainPlan, decisions: JointSamplingRecord) -> None:
    validate_path(plan.operation)
    if plan.version != (
        "native-asymmetric-domain-plan-v1"
        if type(plan) is AsymmetricNativeDomainPlan
        else "native-domain-plan-v1"
    ):
        raise ValueError("unsupported native domain plan version")
    expected: list[
        tuple[str, CategoricalParameter | NumericalParameter | None, str | float, str | None]
    ] = []

    def choice(role: str, choices: tuple[str, ...], value: str) -> None:
        if value not in choices:
            raise ValueError("native domain choice differs from its prior")
        request = CategoricalParameter(f"native_{role}", choices) if len(choices) > 1 else None
        expected.append((role, request, value, "configured-singleton" if request is None else None))

    def number(role: str, bounds: tuple[float, float], value: float) -> None:
        if bounds[0] == bounds[1]:
            if value != bounds[0]:
                raise ValueError("fixed native numerical value differs")
            expected.append((role, None, value, "configured-constant"))
            return
        expected.append(
            (role, NumericalParameter(f"native_{role}", *bounds, scale="log"), value, None)
        )

    choice("kind", plan.prior.kinds, plan.kind)
    if plan.kind == "continuous":
        if (
            plan.cardinality,
            plan.emission,
            plan.frequency_law,
            plan.concentration,
            plan.temperature,
            plan.threshold_law,
            plan.threshold_scale,
        ) != (0, "identity", "uniform", 1.0, 1.0, "none", 1.0):
            raise ValueError("continuous native plan cannot own discrete decisions")
    else:
        if plan.kind == "binary":
            if plan.cardinality != 2:
                raise ValueError("binary native plan requires two categories")
            expected.append(("cardinality", None, "2", "binary-domain-cardinality"))
        else:
            choice(
                "cardinality",
                tuple(str(k) for k in plan.prior.cardinalities),
                str(plan.cardinality),
            )
        choice("emission", plan.prior.emissions, plan.emission)
        choice("frequency_law", plan.prior.frequency_laws, plan.frequency_law)
        if plan.frequency_law == "asymmetric-dirichlet":
            if (
                type(plan) is not AsymmetricNativeDomainPlan
                or type(plan.prior) is not AsymmetricNativeDomainPrior
                or plan.concentrations != plan.prior.concentrations(plan.cardinality)
                or plan.concentration != 1.0
            ):
                raise ValueError(
                    "asymmetric concentration vector differs from owned cardinality prior"
                )
        elif type(plan) is AsymmetricNativeDomainPlan:
            raise ValueError("asymmetric plan requires its explicit frequency law")
        elif plan.frequency_law != "uniform":
            number("concentration", plan.prior.concentration, plan.concentration)
        elif plan.concentration != 1.0:
            raise ValueError("uniform frequencies do not own concentration")
        if plan.emission == "softmax":
            number("temperature", plan.prior.temperature, plan.temperature)
        elif plan.temperature != 1.0:
            raise ValueError("deterministic native emission does not own temperature")
        if plan.emission == "threshold":
            choice("threshold_law", plan.prior.threshold_laws, plan.threshold_law)
            number("threshold_scale", plan.prior.threshold_scale, plan.threshold_scale)
        elif plan.threshold_law != "none" or plan.threshold_scale != 1.0:
            raise ValueError("native score emission does not own cutpoint laws")
    if type(plan) is AsymmetricNativeDomainPlan:
        expected.append(("concentrations", None, "", "configured-cardinality-vector"))
    events = operation_events(decisions, plan.operation)
    if len(events) != len(expected):
        raise ValueError("native domain decision count differs from its owned plan")
    for event, (role, request, value, reason) in zip(events, expected, strict=True):
        if event.operation != plan.operation or event.role != role:
            raise ValueError("native domain named decision ownership differs")
        if isinstance(request, CategoricalParameter):
            if categorical_binding(event, decisions, request).value != value:
                raise ValueError("native categorical decision differs from retained plan")
        elif isinstance(request, NumericalParameter):
            if numerical_binding(event, decisions, request).values != (value,):
                raise ValueError("native numerical decision differs from retained plan")
        elif event.kind != "forced_identity" or event.reason != reason:
            raise ValueError("native fixed decision differs from retained reason")
