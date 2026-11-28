"""Shared source numerical converter choices, including explicit target restrictions."""

from dataclasses import dataclass
from typing import Any

from r0.modules.priors.lib.activation_sampling import finite_number
from r0.modules.priors.lib.categorical_preferences import CategoricalParameter
from r0.modules.priors.lib.joint_activation_plans import (
    JointActivationPlan,
    joint_activation_plan_from_dict,
    sample_joint_activation,
)
from r0.modules.priors.lib.joint_plan_bindings import (
    categorical_binding,
    numerical_binding,
    operation_events,
)
from r0.modules.priors.lib.joint_sampling import JointSamplingContext
from r0.modules.priors.lib.joint_sampling_records import (
    JointSamplingRecord,
    validate_name,
    validate_path,
)
from r0.modules.priors.lib.parameter_preferences import NumericalParameter


@dataclass(frozen=True, slots=True)
class JointNumericalConverterPrior:
    allow_kumaraswamy: bool = True
    allow_activation: bool = True
    disallow_target_warping: bool = False
    variant: str = "readout-only"
    version: str = "joint-numerical-converter-prior-v1"

    def __post_init__(self) -> None:
        if self.version != "joint-numerical-converter-prior-v1" or any(
            type(flag) is not bool
            for flag in (
                self.allow_kumaraswamy,
                self.allow_activation,
                self.disallow_target_warping,
            )
        ):
            raise ValueError("converter prior requires its version and boolean switches")
        if self.variant not in ("historical-tuple", "historical-graph", "readout-only"):
            raise ValueError("unsupported converter plan variant")

    @property
    def modes(self) -> tuple[str, ...]:
        return (
            ("id",)
            + (("kumar",) if self.allow_kumaraswamy else ())
            + (("act",) if self.allow_activation else ())
        )

    def request(self) -> CategoricalParameter:
        return CategoricalParameter("num_conv_mode", self.modes)


_DEFAULT_PRIOR = JointNumericalConverterPrior()


def shape_request(name: str) -> NumericalParameter:
    if name not in ("a", "b"):
        raise ValueError("Kumaraswamy shape name must be a or b")
    return NumericalParameter("kumaraswamy_" + name, 0.2, 5.0, scale="log")


def _target(channel: int, group: str, prior: JointNumericalConverterPrior) -> None:
    if type(channel) is not int or not 0 <= channel < 2**63:
        raise ValueError("converter channel must lie in [0,2**63)")
    validate_name(group, "feature group")
    if type(prior) is not JointNumericalConverterPrior:
        raise ValueError("converter plan requires its frozen prior")


@dataclass(frozen=True, slots=True)
class JointConverterPlan:
    operation: tuple[str, ...]
    channel: int
    feature_group: str
    prior: JointNumericalConverterPrior
    mode: str
    shapes: tuple[float, float] | None = None
    activation: JointActivationPlan | None = None
    version: str = "joint-converter-plan-v1"

    def __post_init__(self) -> None:
        validate_path(self.operation)
        _target(self.channel, self.feature_group, self.prior)
        if self.version != "joint-converter-plan-v1" or self.mode not in self.prior.modes:
            raise ValueError("converter plan requires its version and an enabled mode")
        if self.mode == "kumar":
            if (
                type(self.shapes) is not tuple
                or len(self.shapes) != 2
                or any(not finite_number(v) or not 0.2 <= v <= 5.0 for v in self.shapes)
            ):
                raise ValueError("converter shapes must contain finite a/b in [0.2,5]")
        elif self.shapes is not None:
            raise ValueError("only Kumaraswamy converters can own shape parameters")
        if self.mode == "act":
            if (
                type(self.activation) is not JointActivationPlan
                or self.activation.selector != "standard"
            ):
                raise ValueError("activation converter requires a standard activation plan")
            if self.activation.operation != (*self.operation, "activation"):
                raise ValueError("converter activation must belong to its activation child")
        elif self.activation is not None:
            raise ValueError("only activation converters can own an activation plan")

    @property
    def parameter_count(self) -> int:
        return (
            1
            + len(self.shapes or ())
            + (self.activation.parameter_count if self.activation is not None else 0)
        )

    def validate(self, decisions: JointSamplingRecord) -> None:
        events = operation_events(decisions, self.operation)
        if not events or events[0].operation != self.operation or events[0].role != "mode":
            raise ValueError("converter plan requires its first owned mode decision")
        if self.feature_group == "y" and self.prior.disallow_target_warping:
            if (
                len(events) != 1
                or self.mode != "id"
                or events[0].kind != "forced_identity"
                or events[0].reason != "target-warp-disabled"
            ):
                raise ValueError("restricted target converter requires only its forced identity")
            return
        mode = categorical_binding(events[0], decisions, self.prior.request())
        if self.mode != mode.value:
            raise ValueError("converter mode differs from its recorded realization")
        if self.activation is not None:
            self.activation.validate(decisions)
            if events[1:] != operation_events(decisions, self.activation.operation):
                raise ValueError("converter must own only its selected activation branch")
        else:
            roles = ("mode", "a", "b") if self.mode == "kumar" else ("mode",)
            if tuple(e.role for e in events) != roles or any(
                e.operation != self.operation for e in events
            ):
                raise ValueError("converter must own exactly its selected branch decisions")
            if self.shapes is not None:
                expected = tuple(
                    numerical_binding(e, decisions, shape_request(name)).values[0]
                    for e, name in zip(events[1:], ("a", "b"), strict=True)
                )
                if self.shapes != expected:
                    raise ValueError("converter shapes differ from their recorded realizations")


def sample_joint_converter(
    context: JointSamplingContext,
    name: str,
    *,
    channel: int,
    feature_group: str,
    prior: JointNumericalConverterPrior = _DEFAULT_PRIOR,
) -> JointConverterPlan:
    if type(context) is not JointSamplingContext:
        raise ValueError("joint converter sampling requires a shared context")
    _target(channel, feature_group, prior)
    with context.transaction(name) as operation:
        if feature_group == "y" and prior.disallow_target_warping:
            operation.forced_identity("mode", reason="target-warp-disabled")
            mode = "id"
        else:
            mode = operation.categorical("mode", prior.request()).value
        shapes = None
        activation = None
        if mode == "kumar":
            shapes = (
                operation.numerical("a", shape_request("a")).values[0],
                operation.numerical("b", shape_request("b")).values[0],
            )
        elif mode == "act":
            activation = sample_joint_activation(operation, "activation", selector="standard")
        plan = JointConverterPlan(
            operation.operation, channel, feature_group, prior, mode, shapes, activation
        )
        plan.validate(operation.freeze())
    return plan


def joint_converter_plan_from_dict(
    data: dict[str, Any], *, decisions: JointSamplingRecord
) -> JointConverterPlan:
    fields = dict(data)
    path = fields["operation"]
    if type(path) not in (list, tuple) or not 1 <= len(path) <= 16:
        raise ValueError("converter operation path must be a bounded array")
    fields["operation"] = tuple(path)
    fields["prior"] = JointNumericalConverterPrior(**fields["prior"])
    if fields.get("shapes") is not None:
        shapes = fields["shapes"]
        if type(shapes) not in (list, tuple) or len(shapes) != 2:
            raise ValueError("converter shapes must be an a/b array")
        fields["shapes"] = tuple(shapes)
    if fields.get("activation") is not None:
        fields["activation"] = joint_activation_plan_from_dict(
            fields["activation"], decisions=decisions
        )
    plan = JointConverterPlan(**fields)
    plan.validate(decisions)
    return plan
