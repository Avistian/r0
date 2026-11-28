"""Explicit activation descriptors for scalar and vector causal mechanisms."""

import math
from collections.abc import Callable
from dataclasses import dataclass

import torch
from torch.nn import functional as F

from r0.modules.priors.lib.tabicl_fourier import (
    FOURIER_KINDS,
    evaluate_fourier_activation,
    validate_fourier_parameters,
)
from r0.modules.priors.lib.tabicl_random_scale import (
    RANDOM_SCALE_KIND,
    evaluate_random_scale,
    validate_random_scale,
)

_FIXED: dict[str, Callable[[torch.Tensor], torch.Tensor]] = {
    "identity": lambda x: x,
    "relu": torch.relu,
    "tanh": torch.tanh,
    "sigmoid": torch.sigmoid,
    "selu": F.selu,
    "silu": F.silu,
    "softplus": lambda x: torch.logaddexp(torch.zeros_like(x), x),
    "smooth_relu": lambda x: torch.logaddexp(torch.zeros_like(x), x),
    "relu6": F.relu6,
    "hardtanh": F.hardtanh,
    "sign": torch.sign,
    "sign_positive": lambda x: torch.where(x >= 0, 1.0, -1.0).to(x.dtype),
    "gaussian_bump": lambda x: torch.exp(-x.square()),
    "exp": torch.exp,
    "sqrt_abs": lambda x: x.abs().sqrt(),
    "interval_abs_le_1": lambda x: (x.abs() <= 1).to(x.dtype),
    "interval_0_1": lambda x: ((x >= 0) & (x <= 1)).to(x.dtype),
    "sin": torch.sin,
    "square": torch.square,
    "abs": torch.abs,
    "log": torch.log,
    "log_sigmoid": F.logsigmoid,
    "round": torch.round,
    "step": lambda x: (x > 0).to(x.dtype),
    "heaviside": lambda x: (x > 0).to(x.dtype),
    "heaviside_positive": lambda x: (x >= 0).to(x.dtype),
    "gelu": F.gelu,
    "softmax": lambda x: torch.softmax(x, dim=-1),
    "tied_max": lambda x: (x == x.max(dim=-1, keepdim=True).values).to(x.dtype),
    "argmax": lambda x: F.one_hot(x.argmax(dim=-1), num_classes=x.shape[-1]).to(x.dtype),
    "rank": lambda x: x.argsort(dim=-1, stable=True).argsort(dim=-1, stable=True).to(x.dtype),
    "argsort": lambda x: x.argsort(dim=-1, stable=True).to(x.dtype),
}

_COMMON = {"input_scale", "input_shift", "output_scale", "output_shift"}
_PARAMETERS = {
    "leaky_relu": {"negative_slope"},
    "elu": {"alpha"},
    "softplus": {"beta"},
    "smooth_relu": {"beta"},
    "log_abs": {"epsilon"},
    "modulo": {"period"},
    "power": {"exponent"},
    "integer_power": {"exponent"},
    "relu_power": {"exponent"},
    "signed_power": {"exponent"},
    "reciprocal_abs_power": {"exponent", "epsilon"},
    "log_clip_abs": {"epsilon"},
}


def _finite_parameter(value: float) -> bool:
    try:
        return math.isfinite(value)
    except OverflowError:
        return False


def _log_absolute_sum(value: torch.Tensor, epsilon: float) -> torch.Tensor:
    """Keep log(abs(x)+epsilon) finite even when its intermediate sum overflows."""
    return torch.logaddexp(value.abs().log(), torch.full_like(value, math.log(epsilon)))


@dataclass(frozen=True, slots=True)
class CausalActivation:
    kind: str = "identity"
    parameters: tuple[tuple[str, float], ...] = ()

    @property
    def parameter_count(self) -> int:
        return len(self.parameters)

    @property
    def working_extra_channels(self) -> int:
        return 0

    def validate_width(self, width: int) -> None:
        """Validate fixed channel geometry when a source stage declares it."""
        if self.kind == RANDOM_SCALE_KIND and dict(self.parameters)["channels"] != width:
            raise ValueError("random scale width differs from declared channels")

    def boundary_diagnostic(self, value: torch.Tensor) -> dict[str, object] | None:
        return None

    def diagnostic_input(self, value: torch.Tensor) -> torch.Tensor:
        return value

    def apply(self, value: torch.Tensor) -> torch.Tensor:
        return primitive_activation(value, self)

    @property
    def mixes_channels(self) -> bool:
        return self.kind in ("softmax", "tied_max", "argmax", "rank", "argsort")

    def uses_input(self, width: int) -> bool:
        parameters = dict(self.parameters)
        return (
            parameters.get("input_scale", 1) != 0
            and parameters.get("output_scale", 1) != 0
            and not (self.mixes_channels and width == 1)
            and not (self.kind in ("power", "integer_power") and parameters.get("exponent", 2) == 0)
        )

    def __post_init__(self) -> None:
        if type(
            self.kind
        ) is not str or self.kind not in _FIXED.keys() | _PARAMETERS.keys() | FOURIER_KINDS | {
            RANDOM_SCALE_KIND
        }:
            raise ValueError("unsupported causal activation")
        if type(self.parameters) is not tuple or any(
            type(pair) is not tuple
            or len(pair) != 2
            or type(pair[0]) is not str
            or type(pair[1]) not in (float, int)
            or not _finite_parameter(pair[1])
            for pair in self.parameters
        ):
            raise ValueError("activation parameters must be named finite numeric pairs")
        names = [name for name, _ in self.parameters]
        if self.kind == RANDOM_SCALE_KIND:
            if len(set(names)) != len(names):
                raise ValueError("duplicate random scale parameter")
            validate_random_scale(dict(self.parameters))
            return
        if self.kind in FOURIER_KINDS:
            if len(set(names)) != len(names):
                raise ValueError("duplicate Fourier parameter")
            validate_fourier_parameters(self.kind, dict(self.parameters))
            return
        if len(set(names)) != len(names) or not set(names) <= _COMMON | _PARAMETERS.get(
            self.kind, set()
        ):
            raise ValueError("duplicate or inapplicable activation parameter")
        if any(
            value <= 0 for name, value in self.parameters if name in ("beta", "epsilon", "period")
        ):
            raise ValueError("activation beta, epsilon and period must be positive")
        exponent = dict(self.parameters).get("exponent", 2.0)
        if self.kind == "integer_power" and not float(exponent).is_integer():
            raise ValueError("integer_power requires an integer exponent")
        if self.kind in ("relu_power", "signed_power", "reciprocal_abs_power") and exponent <= 0:
            raise ValueError(
                "ReLU, signed and reciprocal absolute powers require positive exponents"
            )


def activate(value: torch.Tensor, activation: CausalActivation) -> torch.Tensor:
    return activation.apply(value)


def primitive_activation(value: torch.Tensor, activation: CausalActivation) -> torch.Tensor:
    """Apply a declared activation without hiding nonfinite inputs or outputs."""
    if not torch.isfinite(value).all():
        raise ValueError("nonfinite causal activation input; task was not retried")
    parameters = dict(activation.parameters)
    if "input_scale" in parameters or "input_shift" in parameters:
        value = parameters.get("input_scale", 1.0) * value + parameters.get("input_shift", 0.0)
        if not torch.isfinite(value).all():
            raise ValueError("nonfinite rescaled activation input; task was not retried")
    if activation.kind == "log" and torch.any(value <= 0):
        raise ValueError("log activation requires strictly positive inputs; use log_abs explicitly")
    if activation.kind == RANDOM_SCALE_KIND:
        result = evaluate_random_scale(value, parameters)
    elif activation.kind in FOURIER_KINDS:
        result = evaluate_fourier_activation(value, parameters)
    elif activation.kind in ("power", "integer_power"):
        exponent = parameters.get("exponent", 2.0)
        if not float(exponent).is_integer() and torch.any(value < 0):
            raise ValueError(
                "fractional power requires nonnegative inputs; use signed_power explicitly"
            )
        if exponent < 0 and torch.any(value == 0):
            raise ValueError("negative power requires nonzero inputs")
        if float(exponent).is_integer():
            # Torch's floating kernel can round an integer exponent past 2**53.
            # Its magnitude may use float64; parity must use the retained integer.
            result = value.abs().pow(float(exponent))
            if int(exponent) % 2:
                result = torch.where(torch.signbit(value), -result, result)
        else:
            result = value.pow(exponent)
    elif activation.kind == "relu_power":
        result = torch.relu(value).pow(parameters.get("exponent", 2.0))
    elif activation.kind == "signed_power":
        result = value.sign() * value.abs().pow(parameters.get("exponent", 2.0))
    elif activation.kind == "reciprocal_abs_power":
        log_sum = _log_absolute_sum(value, parameters.get("epsilon", 1e-3))
        result = torch.exp(-float(parameters.get("exponent", 2.0)) * log_sum)
    elif activation.kind == "log_clip_abs":
        result = value.abs().clamp_min(parameters.get("epsilon", 1e-6)).log()
    elif activation.kind == "leaky_relu":
        result = F.leaky_relu(value, negative_slope=parameters.get("negative_slope", 0.01))
    elif activation.kind == "elu":
        result = F.elu(value, alpha=parameters.get("alpha", 1.0))
    elif activation.kind in ("softplus", "smooth_relu") and "beta" in parameters:
        beta = parameters["beta"]
        result = torch.clamp_min(value, 0) + torch.log1p(torch.exp(-(beta * value).abs())) / beta
    elif activation.kind == "log_abs":
        result = _log_absolute_sum(value, parameters.get("epsilon", 1e-6))
    elif activation.kind == "modulo":
        result = torch.remainder(value, parameters.get("period", 1.0))
    else:
        result = _FIXED[activation.kind](value)
    if not torch.isfinite(result).all():
        raise ValueError(f"nonfinite {activation.kind} activation output; task was not retried")
    if "output_scale" in parameters or "output_shift" in parameters:
        result = parameters.get("output_scale", 1.0) * result + parameters.get("output_shift", 0.0)
        if not torch.isfinite(result).all():
            raise ValueError("nonfinite rescaled activation output; task was not retried")
    return result


def activation_diagnostic(
    value: torch.Tensor, activation: CausalActivation
) -> dict[str, object] | None:
    """Describe base-activation boundaries, never filter a draw or fit a parameter."""
    custom = activation.boundary_diagnostic(value)
    if custom is not None:
        return custom
    value = activation.diagnostic_input(value)
    parameters = dict(activation.parameters)
    argument = parameters.get("input_scale", 1.0) * value + parameters.get("input_shift", 0.0)
    kind = activation.kind
    flags: list[str] = []
    if parameters.get("input_scale", 1) == 0 or parameters.get("output_scale", 1) == 0:
        flags.append("constant_activation_transform")
    if kind == "tanh":
        boundary = argument.tanh().abs() >= 1 - 1e-6
    elif kind == "sigmoid":
        sigmoid = argument.sigmoid()
        boundary = (sigmoid <= 1e-6) | (sigmoid >= 1 - 1e-6)
    elif kind == "hardtanh":
        boundary = argument.abs() >= 1
    elif kind == "relu6":
        boundary = (argument <= 0) | (argument >= 6)
    elif kind == "relu":
        boundary = argument <= 0
    elif flags:
        boundary = torch.ones_like(argument, dtype=torch.bool)
    else:
        return None
    fractions: list[float] = boundary.to(torch.float64).mean(dim=0).detach().cpu().numpy().tolist()
    if kind in ("tanh", "sigmoid", "hardtanh", "relu6", "relu") and any(
        f >= 0.95 for f in fractions
    ):
        flags.append("inactive_relu" if kind == "relu" else "saturated_activation")
    return {
        "kind": kind,
        "rows": len(value),
        "boundary_fraction": fractions,
        "fraction_threshold": 0.95,
        "tail_tolerance": 1e-6,
        "flags": flags,
        "meaning": (
            "base activation after input transform; before output scaling and any later noise"
        ),
    }
