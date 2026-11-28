"""Source activation decisions bound to one operation in the shared context."""

from dataclasses import dataclass
from typing import Any

from r0.modules.priors.lib.activation_sampling import finite_number
from r0.modules.priors.lib.causal_activations import CausalActivation
from r0.modules.priors.lib.joint_plan_bindings import (
    categorical_binding,
    numerical_binding,
    operation_events,
)
from r0.modules.priors.lib.joint_sampling import JointSamplingContext
from r0.modules.priors.lib.joint_sampling_records import (
    JointSamplingRecord,
    validate_path,
)
from r0.modules.priors.lib.source_activation_law import (
    exponent_request,
    fixed_request,
    selector_request,
)
from r0.modules.priors.lib.source_causal_prior import rescale_request


@dataclass(frozen=True, slots=True)
class JointActivationPlan:
    operation: tuple[str, ...]
    selector: str
    activation: CausalActivation
    scale: float | None = None
    version: str = "joint-activation-plan-v1"

    def __post_init__(self) -> None:
        validate_path(self.operation)
        selector_request(self.selector)
        if (
            self.version != "joint-activation-plan-v1"
            or type(self.activation) is not CausalActivation
        ):
            raise ValueError("joint activation plan requires its version and a frozen primitive")
        if self.selector == "plain":
            if self.scale is not None:
                raise ValueError("plain activation plan cannot own a standard rescale")
        elif self.scale is None or not finite_number(self.scale) or not 1 <= self.scale <= 10:
            raise ValueError("standard activation plan requires a finite scale in [1,10]")

    @property
    def parameter_count(self) -> int:
        """Realized numeric fields in this plan, in addition to the shared ledger."""
        return self.activation.parameter_count + int(self.scale is not None)

    def validate(self, decisions: JointSamplingRecord) -> None:
        events = operation_events(decisions, self.operation)
        if not events or any(e.operation != self.operation for e in events):
            raise ValueError("activation operation must own its leaf decisions")
        slot = categorical_binding(events[0], decisions, selector_request(self.selector))
        branch = "primitive" if slot.value.startswith("fixed-") else "exponent"
        expected_roles = (
            ("selector", "scale", branch) if self.selector == "standard" else ("selector", branch)
        )
        if tuple(e.role for e in events) != expected_roles:
            raise ValueError("activation operation must own exactly its ordered branch decisions")
        if self.selector == "standard":
            scale = numerical_binding(events[1], decisions, rescale_request()).values[0]
            if self.scale != scale:
                raise ValueError("activation scale differs from its recorded realization")
        if branch == "primitive":
            primitive = categorical_binding(events[-1], decisions, fixed_request())
            expected = CausalActivation(primitive.value)
        else:
            exponent = numerical_binding(
                events[-1], decisions, exponent_request(slot.value)
            ).values[0]
            expected = CausalActivation(slot.value, (("exponent", exponent),))
        if self.activation != expected:
            raise ValueError("activation primitive differs from its recorded realization")


def sample_joint_activation(
    context: JointSamplingContext, name: str, *, selector: str = "standard"
) -> JointActivationPlan:
    """Freeze source choices; fitting of standard wrappers happens later on ancestors."""
    if type(context) is not JointSamplingContext:
        raise ValueError("joint activation sampling requires a shared context")
    request = selector_request(selector)
    with context.transaction(name) as operation:
        slot = operation.categorical("selector", request)
        # In the lazy source wrapper, rescaling fits before the chosen primitive.
        scale = (
            operation.numerical("scale", rescale_request()).values[0]
            if selector == "standard"
            else None
        )
        if slot.value.startswith("fixed-"):
            primitive = operation.categorical("primitive", fixed_request())
            activation = CausalActivation(primitive.value)
        else:
            exponent = operation.numerical("exponent", exponent_request(slot.value)).values[0]
            activation = CausalActivation(slot.value, (("exponent", exponent),))
        plan = JointActivationPlan(operation.operation, selector, activation, scale)
        plan.validate(operation.freeze())
    return plan


def joint_activation_plan_from_dict(
    data: dict[str, Any], *, decisions: JointSamplingRecord
) -> JointActivationPlan:
    fields = dict(data)
    path = fields["operation"]
    if type(path) not in (list, tuple) or not 1 <= len(path) <= 16:
        raise ValueError("activation operation path must be a bounded array")
    fields["operation"] = tuple(path)
    activation = dict(fields["activation"])
    parameters = activation.get("parameters", ())
    if type(parameters) not in (list, tuple) or len(parameters) > 1:
        raise ValueError("source activation plan permits at most one primitive parameter")
    if any(type(pair) not in (list, tuple) or len(pair) != 2 for pair in parameters):
        raise ValueError("activation parameter pair must be a two-element array")
    activation["parameters"] = tuple(tuple(p) for p in parameters)
    fields["activation"] = CausalActivation(**activation)
    plan = JointActivationPlan(**fields)
    plan.validate(decisions)
    return plan
