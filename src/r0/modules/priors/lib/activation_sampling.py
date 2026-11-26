"""Frozen activation preference specifications, independent of generated rows."""

from dataclasses import dataclass

from r0.modules.priors.lib.causal_activations import CausalActivation


@dataclass(frozen=True, slots=True)
class CausalActivationPrior:
    choices: tuple[CausalActivation, ...] = (CausalActivation(),)
    weights: tuple[float, ...] = ()
    mode: str = "independent"
    concentration: float = 1.0
    placement: str = "output"
    noise_after_probability: float = 0.5
    version: str = "activation-prior-v1"
