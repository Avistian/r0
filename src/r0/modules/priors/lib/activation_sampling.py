"""Frozen activation preference specifications, independent of generated rows."""

import math
from dataclasses import dataclass
from typing import Any

import numpy as np

from r0.modules.priors.lib.causal_activations import CausalActivation


def finite_number(value: float) -> bool:
    try:
        return type(value) in (float, int) and math.isfinite(value)
    except OverflowError:
        return False


@dataclass(frozen=True, slots=True)
class ActivationParameterPrior:
    name: str
    low: float
    high: float
    distribution: str = "uniform"

    def __post_init__(self) -> None:
        if type(self.name) is not str or not self.name:
            raise ValueError("activation parameter name must be a nonempty string")
        if not finite_number(self.low) or not finite_number(self.high) or self.low > self.high:
            raise ValueError("activation parameter bounds must be ordered finite numbers")
        if self.distribution not in ("uniform", "log_uniform", "randint"):
            raise ValueError("unsupported activation parameter distribution")
        if self.distribution == "log_uniform" and self.low <= 0:
            raise ValueError("log-uniform parameters require positive bounds")
        if self.distribution == "randint" and (
            type(self.low) is not int
            or type(self.high) is not int
            or not -(2**63) <= self.low <= self.high < 2**63
        ):
            raise ValueError("integer parameter bounds must be signed 64-bit integers")

    def draw(self, rng: np.random.Generator) -> float:
        if self.low == self.high:
            return self.low
        if self.distribution == "randint":
            return int(rng.integers(int(self.low), int(self.high), endpoint=True))
        fraction = float(rng.random())
        if self.distribution == "log_uniform":
            return math.exp((1 - fraction) * math.log(self.low) + fraction * math.log(self.high))
        return (1 - fraction) * self.low + fraction * self.high

    def contains(self, value: float) -> bool:
        return self.low <= value <= self.high and (
            self.distribution != "randint" or type(value) is int
        )


@dataclass(frozen=True, slots=True)
class CausalActivationPrior:
    choices: tuple[CausalActivation, ...] = (CausalActivation(),)
    weights: tuple[float, ...] = ()
    mode: str = "independent"
    concentration: float = 1.0
    placement: str = "output"
    noise_after_probability: float = 0.5
    version: str = "activation-prior-v1"
    parameter_priors: tuple[tuple[ActivationParameterPrior, ...], ...] = ()

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

        if (
            type(self.parameter_priors) is not tuple
            or self.parameter_priors
            and (
                len(self.parameter_priors) != len(self.choices)
                or any(
                    type(group) is not tuple
                    or any(type(p) is not ActivationParameterPrior for p in group)
                    for group in self.parameter_priors
                )
            )
        ):
            raise ValueError("parameter priors must match activation choices")
        for choice, group in zip(self.choices, self.parameter_priors, strict=False):
            names = [p.name for p in group]
            if len(set(names)) != len(names) or set(names) & dict(choice.parameters).keys():
                raise ValueError("parameter priors cannot duplicate frozen or sampled parameters")
            for endpoint in ("low", "high"):
                CausalActivation(
                    choice.kind,
                    choice.parameters + tuple((p.name, getattr(p, endpoint)) for p in group),
                )
            if choice.kind == "integer_power" and any(
                p.name == "exponent" and p.low != p.high and p.distribution != "randint"
                for p in group
            ):
                raise ValueError("integer powers require an integer exponent distribution")

    def sample_choice(self, index: int, rng: np.random.Generator) -> CausalActivation:
        choice = self.choices[index]
        group = self.parameter_priors[index] if self.parameter_priors else ()
        return CausalActivation(
            choice.kind, choice.parameters + tuple((p.name, p.draw(rng)) for p in group)
        )

    def accepts(self, index: int, value: CausalActivation) -> bool:
        choice = self.choices[index]
        group = self.parameter_priors[index] if self.parameter_priors else ()
        actual, frozen = dict(value.parameters), dict(choice.parameters)
        return (
            value.kind == choice.kind
            and actual.keys() == frozen.keys() | {p.name for p in group}
            and all(actual[name] == v for name, v in frozen.items())
            and all(p.contains(actual[p.name]) for p in group)
        )

    def base_probabilities(self) -> tuple[float, ...]:
        weights = self.weights or (1.0,) * len(self.choices)
        scaled = tuple(w / max(weights) for w in weights)
        total = math.fsum(scaled)
        probabilities = tuple(w / total for w in scaled)
        if any(w > 0 and p == 0 for w, p in zip(weights, probabilities, strict=True)):
            raise ValueError("unrepresentable activation probabilities; positive weight underflow")
        return probabilities


def activation_prior_from_dict(data: dict[str, Any]) -> CausalActivationPrior:
    fields = dict(data)
    choices: list[CausalActivation] = []
    for choice in fields["choices"]:
        values = dict(choice)
        values["parameters"] = tuple(tuple(p) for p in values.get("parameters", ()))
        choices.append(CausalActivation(**values))
    fields["choices"] = tuple(choices)
    fields["weights"] = tuple(fields.get("weights", ()))
    fields["parameter_priors"] = tuple(
        tuple(ActivationParameterPrior(**p) for p in group)
        for group in fields.get("parameter_priors", ())
    )
    return CausalActivationPrior(**fields)
