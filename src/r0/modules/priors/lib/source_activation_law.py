"""Versioned source selector slots and retained links to primitive decisions."""

from dataclasses import dataclass

from r0.modules.priors.lib.categorical_preferences import CategoricalParameter
from r0.modules.priors.lib.causal_activations import CausalActivation
from r0.modules.priors.lib.parameter_preferences import NumericalParameter

# Slot identity matters: repeated fixed-family slots have independently weighted masses.
FIXED_ACTIVATIONS = (
    "tanh",
    "leaky_relu",
    "elu",
    "identity",
    "selu",
    "silu",
    "relu",
    "softplus",
    "relu6",
    "hardtanh",
    "sign_positive",
    "heaviside_positive",
    "gaussian_bump",
    "exp",
    "interval_abs_le_1",
    "sin",
    "square",
    "abs",
    "log_clip_abs",
    "rank",
    "sigmoid",
    "round",
    "modulo",
    "softmax",
    "tied_max",
    "argsort",
    "log_sigmoid",
)


def selector_request(selector: str) -> CategoricalParameter:
    if selector not in ("standard", "plain"):
        raise ValueError("source activation selector must be standard or plain")
    repeats = 4 if selector == "standard" else 8
    powers = ("relu_power", "signed_power")
    if selector == "plain":
        powers += ("reciprocal_abs_power", "integer_power")
    return CategoricalParameter(
        "act_type" if selector == "standard" else "act_type_plain",
        tuple(f"fixed-{i}" for i in range(repeats)) + powers,
    )


def fixed_request() -> CategoricalParameter:
    return CategoricalParameter("acts_torch", FIXED_ACTIVATIONS)


def exponent_request(kind: str) -> NumericalParameter:
    names = {
        "relu_power": "relu_random_power_exponent",
        "signed_power": "random_power_exponent",
        "reciprocal_abs_power": "neg_random_power_exponent",
        "integer_power": "rand_int_power_exponent",
    }
    if kind not in names:
        raise ValueError("source activation exponent requires a power family")
    if kind == "integer_power":
        return NumericalParameter(names[kind], 2, 6, integer="floor")
    return NumericalParameter(names[kind], 0.1, 10.0, scale="log")


@dataclass(frozen=True, slots=True)
class SourceActivationSelection:
    selector: str
    activation: CausalActivation
    slot_draw: int
    primitive_draw: int | None = None
    exponent_draw: int | None = None

    def __post_init__(self) -> None:
        selector_request(self.selector)
        if type(self.activation) is not CausalActivation:
            raise ValueError("source selection requires a frozen primitive activation")
        if (
            type(self.slot_draw) is not int
            or self.slot_draw < 0
            or any(
                index is not None and (type(index) is not int or index < 0)
                for index in (self.primitive_draw, self.exponent_draw)
            )
        ):
            raise ValueError("source activation selection requires nonnegative draw references")
        if (self.primitive_draw is None) == (self.exponent_draw is None):
            raise ValueError("source selection binds exactly one primitive or exponent draw")

    @property
    def parameter_count(self) -> int:
        return 2 + len(self.activation.parameters)
