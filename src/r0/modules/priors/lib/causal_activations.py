"""Explicit activation descriptors for scalar and vector causal mechanisms."""

from dataclasses import dataclass

import torch


@dataclass(frozen=True, slots=True)
class CausalActivation:
    kind: str = "identity"

    def __post_init__(self) -> None:
        if self.kind not in ("identity", "relu"):
            raise ValueError("unsupported causal activation")


def activate(value: torch.Tensor, activation: CausalActivation) -> torch.Tensor:
    """Apply a declared activation without hiding nonfinite inputs or outputs."""
    if not torch.isfinite(value).all():
        raise ValueError("nonfinite causal activation input; task was not retried")
    return torch.relu(value) if activation.kind == "relu" else value
