"""Explicit activation descriptors for scalar and vector causal mechanisms."""

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


@dataclass(frozen=True, slots=True)
class CausalActivation:
    kind: str = "identity"
    parameters: tuple[tuple[str, float], ...] = ()

    def __post_init__(self) -> None:
        if type(self.kind) is not str or self.kind not in _FIXED:
            raise ValueError("unsupported causal activation")


def activate(value: torch.Tensor, activation: CausalActivation) -> torch.Tensor:
    """Apply a declared activation without hiding nonfinite inputs or outputs."""
    if not torch.isfinite(value).all():
        raise ValueError("nonfinite causal activation input; task was not retried")
    if activation.kind == "log" and torch.any(value <= 0):
        raise ValueError("log activation requires strictly positive inputs; use log_abs explicitly")
    result = _FIXED[activation.kind](value)
    if not torch.isfinite(result).all():
        raise ValueError(f"nonfinite {activation.kind} activation output; task was not retried")
    return result
