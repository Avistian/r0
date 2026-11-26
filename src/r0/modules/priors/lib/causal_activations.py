"""Explicit activation descriptors for scalar and vector causal mechanisms."""

import math
from collections.abc import Callable
from dataclasses import dataclass

import torch
from torch.nn import functional as F

_FIXED: dict[str, Callable[[torch.Tensor], torch.Tensor]] = {
    "identity": lambda x: x,
    "relu": torch.relu,
    "tanh": torch.tanh,
    "sigmoid": torch.sigmoid,
    "leaky_relu": F.leaky_relu,
    "elu": F.elu,
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
    "log_abs": lambda x: torch.log(x.abs() + 1e-6),
    "log_sigmoid": F.logsigmoid,
    "round": torch.round,
    "step": lambda x: (x > 0).to(x.dtype),
    "heaviside": lambda x: (x > 0).to(x.dtype),
    "heaviside_positive": lambda x: (x >= 0).to(x.dtype),
    "modulo": lambda x: torch.remainder(x, 1.0),
    "gelu": F.gelu,
}

_COMMON = {"input_scale", "input_shift", "output_scale", "output_shift"}
_PARAMETERS = {
    "leaky_relu": {"negative_slope"},
    "elu": {"alpha"},
    "softplus": {"beta"},
    "smooth_relu": {"beta"},
    "log_abs": {"epsilon"},
    "modulo": {"period"},
}


@dataclass(frozen=True, slots=True)
class CausalActivation:
    kind: str = "identity"
    parameters: tuple[tuple[str, float], ...] = ()

    def __post_init__(self) -> None:
        if type(self.kind) is not str or self.kind not in _FIXED:
            raise ValueError("unsupported causal activation")
        if type(self.parameters) is not tuple or any(
            type(pair) is not tuple
            or len(pair) != 2
            or type(pair[0]) is not str
            or type(pair[1]) not in (float, int)
            or not math.isfinite(pair[1])
            for pair in self.parameters
        ):
            raise ValueError("activation parameters must be named finite numeric pairs")
        names = [name for name, _ in self.parameters]
        if len(set(names)) != len(names) or not set(names) <= _COMMON | _PARAMETERS.get(
            self.kind, set()
        ):
            raise ValueError("duplicate or inapplicable activation parameter")
        if any(
            value <= 0 for name, value in self.parameters if name in ("beta", "epsilon", "period")
        ):
            raise ValueError("activation beta, epsilon and period must be positive")


def activate(value: torch.Tensor, activation: CausalActivation) -> torch.Tensor:
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
    if activation.kind == "leaky_relu":
        result = F.leaky_relu(value, negative_slope=parameters.get("negative_slope", 0.01))
    elif activation.kind == "elu":
        result = F.elu(value, alpha=parameters.get("alpha", 1.0))
    elif activation.kind in ("softplus", "smooth_relu") and "beta" in parameters:
        beta = parameters["beta"]
        result = torch.clamp_min(value, 0) + torch.log1p(torch.exp(-(beta * value).abs())) / beta
    elif activation.kind == "log_abs":
        result = torch.log(value.abs() + parameters.get("epsilon", 1e-6))
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
