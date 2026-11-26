"""Causal recipes that retain the activation sampling law and realized preferences."""

import math
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

from r0.modules.priors.lib.activation_sampling import CausalActivationPrior, finite_number
from r0.modules.priors.lib.causal import ActivatedCausalNode, CausalRecipe, MatrixCausalNode
from r0.modules.priors.lib.causal_activations import CausalActivation
from r0.modules.priors.lib.representatives import record_fingerprint
from r0.modules.priors.lib.weight_matrices import FactorizedWeights


@dataclass(frozen=True, slots=True)
class SampledCausalRecipe(CausalRecipe):
    activation_prior: CausalActivationPrior = CausalActivationPrior()
    activation_probabilities: tuple[float, ...] = (1.0,)

    def __post_init__(self) -> None:
        CausalRecipe.__post_init__(self)
        if self.version != "causal-v2" or self.construction not in (
            "gaussian-activation-prior-v1",
            "activation-exogenous-v1",
            "activation-factorized-v1",
        ):
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
        base = self.activation_prior.base_probabilities()
        if any(w == 0 and p != 0 for w, p in zip(base, probabilities, strict=True)):
            raise ValueError("sampling probabilities cannot enable a zero-weight choice")
        if self.activation_prior.mode == "dirichlet" and any(
            w > 0 and p == 0 for w, p in zip(base, probabilities, strict=True)
        ):
            raise ValueError("Dirichlet sampling probabilities must retain positive support")
        if self.activation_prior.mode == "independent" and probabilities != base:
            raise ValueError("independent sampling probabilities must match the prior weights")
        selected = [i for i, p in enumerate(probabilities) if p > 0]
        homogeneous: set[CausalActivation] = set()
        if self.activation_prior.mode == "homogeneous" and (
            len(selected) != 1 or probabilities[selected[0]] != 1.0
        ):
            raise ValueError("homogeneous activation preferences must be one-hot")
        for node in self.nodes:
            if (
                self.construction == "activation-factorized-v1"
                and node.parents
                and (
                    not isinstance(node, MatrixCausalNode)
                    or type(node.weight_construction) is not FactorizedWeights
                )
            ):
                raise ValueError("factorized sampling requires every projection construction")
            if not isinstance(node, ActivatedCausalNode):
                raise ValueError("activation-sampled nodes must record their activation stages")
            after = self.activation_prior.noise_after_probability
            if (after == 0 and node.noise_position != "before_output") or (
                after == 1 and node.noise_position != "after_output"
            ):
                raise ValueError("node noise order disagrees with its sampling law")
            for stage, active in (
                (
                    node.input_activation,
                    bool(node.parents) and self.activation_prior.placement in ("input", "both"),
                ),
                (node.output_activation, self.activation_prior.placement in ("output", "both")),
            ):
                if active:
                    homogeneous.add(stage)
                    if not any(self.activation_prior.accepts(i, stage) for i in selected):
                        raise ValueError("node activation disagrees with recorded sampling support")
                elif stage != CausalActivation():
                    raise ValueError("inactive activation stage must be identity")
        if self.activation_prior.mode == "homogeneous" and len(homogeneous) > 1:
            raise ValueError("homogeneous graphs must share frozen activation parameters")

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
        if not np.isfinite(gamma).all() or np.any(gamma <= 0):
            raise ValueError("nonfinite or underflowed Dirichlet draw; draw was not retried")
        gamma = gamma / gamma.max()
        values = np.zeros(len(base))
        values[support] = gamma / gamma.sum()
        if np.any(values[support] <= 0):
            raise ValueError("underflowed Dirichlet probabilities; draw was not retried")
        probabilities = tuple(float(p) for p in values)

    common = (
        prior.sample_choice(probabilities.index(1.0), rng) if prior.mode == "homogeneous" else None
    )

    def choose() -> CausalActivation:
        return (
            common
            if common is not None
            else prior.sample_choice(int(rng.choice(len(prior.choices), p=probabilities)), rng)
        )

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
