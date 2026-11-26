"""Frozen activation preference specifications, independent of generated rows."""

import math
from dataclasses import dataclass
from typing import Any

from r0.modules.priors.lib.causal_activations import CausalActivation


def finite_number(value: float) -> bool:
    try:
        return type(value) in (float, int) and math.isfinite(value)
    except OverflowError:
        return False


@dataclass(frozen=True, slots=True)
class CausalActivationPrior:
    choices: tuple[CausalActivation, ...] = (CausalActivation(),)
    weights: tuple[float, ...] = ()
    mode: str = "independent"
    concentration: float = 1.0
    placement: str = "output"
    noise_after_probability: float = 0.5
    version: str = "activation-prior-v1"

    def __post_init__(self) -> None:
        if self.version != "activation-prior-v1" or self.mode not in (
            "independent",
            "dirichlet",
            "homogeneous",
        ):
            raise ValueError("unsupported activation prior version or mode")
        if (
            type(self.choices) is not tuple
            or not 1 <= len(self.choices) <= 256
            or any(type(choice) is not CausalActivation for choice in self.choices)
        ):
            raise ValueError("activation prior requires 1..256 frozen choices")
        if (
            type(self.weights) is not tuple
            or self.weights
            and (
                len(self.weights) != len(self.choices)
                or any(not finite_number(w) or w < 0 for w in self.weights)
                or not any(self.weights)
            )
        ):
            raise ValueError("activation weights must be nonnegative with positive total")
        if not finite_number(self.concentration) or self.concentration <= 0:
            raise ValueError("Dirichlet total concentration must be positive and finite")
        if self.mode != "dirichlet" and self.concentration != 1:
            raise ValueError("concentration applies only to Dirichlet preferences")
        if self.placement not in ("input", "output", "both"):
            raise ValueError("activation placement must be input, output or both")
        if (
            not finite_number(self.noise_after_probability)
            or not 0 <= self.noise_after_probability <= 1
        ):
            raise ValueError("noise-after probability must be in [0,1]")

    def base_probabilities(self) -> tuple[float, ...]:
        weights = self.weights or (1.0,) * len(self.choices)
        scaled = tuple(w / max(weights) for w in weights)
        total = math.fsum(scaled)
        return tuple(w / total for w in scaled)


def activation_prior_from_dict(data: dict[str, Any]) -> CausalActivationPrior:
    fields = dict(data)
    choices: list[CausalActivation] = []
    for choice in fields["choices"]:
        values = dict(choice)
        values["parameters"] = tuple(tuple(p) for p in values.get("parameters", ()))
        choices.append(CausalActivation(**values))
    fields["choices"] = tuple(choices)
    fields["weights"] = tuple(fields.get("weights", ()))
    return CausalActivationPrior(**fields)
