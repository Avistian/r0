"""Named latent width decisions plus explicit converter channel reservations."""

from dataclasses import dataclass

from r0.modules.priors.lib.causal_limits import CausalLimits
from r0.modules.priors.lib.joint_plan_bindings import numerical_binding, operation_events
from r0.modules.priors.lib.joint_sampling import JointSamplingContext
from r0.modules.priors.lib.joint_sampling_records import JointSamplingRecord, validate_path
from r0.modules.priors.lib.parameter_preferences import NumericalParameter

_LIMITS = CausalLimits()


@dataclass(frozen=True, slots=True)
class JointWidthPrior:
    low: int = 1
    high: int = 32
    version: str = "joint-width-prior-v1"

    def __post_init__(self) -> None:
        if (
            self.version != "joint-width-prior-v1"
            or type(self.low) is not int
            or type(self.high) is not int
            or not 1 <= self.low < self.high <= 2**53
        ):
            raise ValueError("latent width prior requires positive ordered integer bounds")

    def request(self) -> NumericalParameter:
        return NumericalParameter(
            "node_n_latent_features", self.low, self.high, scale="log", integer="floor"
        )


_PRIOR = JointWidthPrior()


def _admit(converter_channels: tuple[int, ...], limits: CausalLimits) -> None:
    if type(limits) is not CausalLimits:
        raise ValueError("width sampling requires frozen causal limits")
    if (
        type(converter_channels) is not tuple
        or not converter_channels
        or len(converter_channels) > limits.nodes
        or any(type(n) is not int or not 0 <= n < limits.node_channels for n in converter_channels)
    ):
        raise ValueError("converter reservations exceed node or channel limits")
    if sum(converter_channels) + len(converter_channels) > limits.total_channels:
        raise ValueError("converter reservations exceed total channel limit")
    if (
        8 * len(converter_channels) + 128 > limits.parameters
        or 2048 * len(converter_channels) + 8192 > limits.working_bytes
    ):
        raise ValueError("latent width metadata limit exceeded")


@dataclass(frozen=True, slots=True)
class JointWidthPlan:
    operation: tuple[str, ...]
    latent: tuple[int, ...]
    converter_channels: tuple[int, ...]
    prior: JointWidthPrior = _PRIOR
    limits: CausalLimits = _LIMITS
    version: str = "joint-width-plan-v1"

    def __post_init__(self) -> None:
        validate_path(self.operation)
        if self.version != "joint-width-plan-v1" or type(self.prior) is not JointWidthPrior:
            raise ValueError("width plan requires its version and frozen prior")
        _admit(self.converter_channels, self.limits)
        if (
            type(self.latent) is not tuple
            or len(self.latent) != len(self.converter_channels)
            or any(
                type(n) is not int or not self.prior.low <= n < self.prior.high for n in self.latent
            )
        ):
            raise ValueError("latent widths must match node count and prior bounds")
        if (
            any(n > self.limits.node_channels for n in self.widths)
            or sum(self.widths) > self.limits.total_channels
        ):
            raise ValueError("sampled latent widths exceed channel limit")

    @property
    def widths(self) -> tuple[int, ...]:
        return tuple(a + b for a, b in zip(self.latent, self.converter_channels, strict=True))

    def validate(self, decisions: JointSamplingRecord) -> None:
        events = operation_events(decisions, self.operation)
        if len(events) != len(self.latent):
            raise ValueError("width plan must own exactly one decision per node")
        for i, (event, width) in enumerate(zip(events, self.latent, strict=True)):
            if event.operation != self.operation or event.role != f"node:{i}":
                raise ValueError("width plan decision ownership differs")
            draw = numerical_binding(event, decisions, self.prior.request())
            if draw.values != (width,):
                raise ValueError("latent width differs from its named realization")


def sample_joint_widths(
    context: JointSamplingContext,
    name: str,
    *,
    converter_channels: tuple[int, ...],
    prior: JointWidthPrior = _PRIOR,
    limits: CausalLimits = _LIMITS,
) -> JointWidthPlan:
    """Retain unobserved latent channels; reject over-budget draws without retry."""
    if type(context) is not JointSamplingContext or type(prior) is not JointWidthPrior:
        raise ValueError("width sampling requires a shared context and frozen prior")
    _admit(converter_channels, limits)
    with context.transaction(name) as operation:
        latent = tuple(
            int(operation.numerical(f"node:{i}", prior.request()).values[0])
            for i in range(len(converter_channels))
        )
        plan = JointWidthPlan(operation.operation, latent, converter_channels, prior, limits)
        plan.validate(operation.freeze())
        return plan
