"""Causal recipes that retain the activation sampling law and realized preferences."""

import math
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

from r0.modules.priors.lib.activation_sampling import CausalActivationPrior, finite_number
from r0.modules.priors.lib.causal import ActivatedCausalNode, CausalRecipe
from r0.modules.priors.lib.causal_activations import CausalActivation
from r0.modules.priors.lib.representatives import record_fingerprint


@dataclass(frozen=True, slots=True)
class SampledCausalRecipe(CausalRecipe):
    activation_prior: CausalActivationPrior = CausalActivationPrior()
    activation_probabilities: tuple[float, ...] = (1.0,)

    def __post_init__(self) -> None:
        CausalRecipe.__post_init__(self)
        if self.version != "causal-v2" or self.construction != "gaussian-activation-prior-v1":
            raise ValueError("sampled activation recipes require their v2 construction")
        if type(self.activation_prior) is not CausalActivationPrior:
            raise ValueError("activation sampling requires a frozen prior")
        probabilities = self.activation_probabilities
        if (
            type(probabilities) is not tuple
            or len(probabilities) != len(self.activation_prior.choices)
            or any(not finite_number(p) or not 0 <= p <= 1 for p in probabilities)
            or not math.isclose(math.fsum(probabilities), 1, rel_tol=0, abs_tol=1e-12)
        ):
            raise ValueError("activation probabilities must form a simplex matching the choices")
        selected = [
            choice
            for choice, p in zip(self.activation_prior.choices, probabilities, strict=True)
            if p > 0
        ]
        if self.activation_prior.mode == "homogeneous" and len(selected) != 1:
            raise ValueError("homogeneous activation preferences must be one-hot")
        for node in self.nodes:
            if not isinstance(node, ActivatedCausalNode):
                raise ValueError("activation-sampled nodes must record their activation stages")
            for stage, active in (
                (
                    node.input_activation,
                    bool(node.parents) and self.activation_prior.placement in ("input", "both"),
                ),
                (node.output_activation, self.activation_prior.placement in ("output", "both")),
            ):
                if stage not in selected if active else stage != CausalActivation():
                    raise ValueError("node activation disagrees with recorded sampling support")

    @property
    def sampling_record(self) -> dict[str, Any]:
        return {
            "prior": asdict(self.activation_prior),
            "probabilities": self.activation_probabilities,
        }


def with_activation_prior(
    recipe: CausalRecipe, prior: CausalActivationPrior, *, seed: int, task_id: int
) -> SampledCausalRecipe:
    if type(prior) is not CausalActivationPrior:
        raise ValueError("activation sampling requires CausalActivationPrior")
    rng = np.random.default_rng(
        int(record_fingerprint((asdict(prior), seed, task_id, "activation-prior-v1")), 16)
    )
    base = prior.base_probabilities()
    probabilities = base
    if prior.mode == "homogeneous":
        chosen = int(rng.choice(len(prior.choices), p=base))
        probabilities = tuple(float(i == chosen) for i in range(len(prior.choices)))
    elif prior.mode == "dirichlet":
        support = np.asarray(base) > 0
        alpha = prior.concentration * np.asarray(base)[support]
        if not np.isfinite(alpha).all() or np.any(alpha <= 0):
            raise ValueError("unrepresentable Dirichlet parameters; draw was not retried")
        gamma = rng.gamma(alpha)
        if not np.isfinite(gamma).all() or not np.any(gamma > 0):
            raise ValueError("nonfinite or underflowed Dirichlet draw; draw was not retried")
        gamma = gamma / gamma.max()
        values = np.zeros(len(base))
        values[support] = gamma / gamma.sum()
        probabilities = tuple(float(p) for p in values)

    def choose() -> CausalActivation:
        return prior.choices[int(rng.choice(len(prior.choices), p=probabilities))]

    nodes = tuple(
        ActivatedCausalNode(
            **asdict(node),
            input_activation=choose()
            if node.parents and prior.placement in ("input", "both")
            else CausalActivation(),
            output_activation=choose()
            if prior.placement in ("output", "both")
            else CausalActivation(),
            noise_position="after_output"
            if rng.random() < prior.noise_after_probability
            else "before_output",
        )
        for node in recipe.nodes
    )
    return SampledCausalRecipe(
        task=recipe.task,
        nodes=nodes,
        observed=recipe.observed,
        target=recipe.target,
        limits=recipe.limits,
        topology=recipe.topology,
        version="causal-v2",
        construction="gaussian-activation-prior-v1",
        activation_prior=prior,
        activation_probabilities=probabilities,
    )
