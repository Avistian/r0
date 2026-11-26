"""Explicit activation descriptors for scalar and vector causal mechanisms."""

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class CausalActivation:
    kind: str = "identity"
