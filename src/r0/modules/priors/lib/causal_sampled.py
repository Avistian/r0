"""Causal recipes that retain the activation sampling law and realized preferences."""

from dataclasses import dataclass

from r0.modules.priors.lib.activation_sampling import CausalActivationPrior
from r0.modules.priors.lib.causal import CausalRecipe


@dataclass(frozen=True, slots=True)
class SampledCausalRecipe(CausalRecipe):
    activation_prior: CausalActivationPrior = CausalActivationPrior()
    activation_probabilities: tuple[float, ...] = (1.0,)
