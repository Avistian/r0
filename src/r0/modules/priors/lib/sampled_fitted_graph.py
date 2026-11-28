"""Bind sampled fit plans to independently fitted nodes without changing old archives."""

from dataclasses import dataclass

import numpy as np

from r0.modules.priors.lib.fitted_graph import FittedFunctionalCausalRecipe, sample_fitted_graph
from r0.modules.priors.lib.fitted_parameter_prior import FittedTreeParameterPrior
from r0.modules.priors.lib.fitted_parameter_sampling import (
    SampledFittedTreePlan,
    sample_fitted_tree_plan,
)
from r0.modules.priors.lib.functional_causal import FunctionalCausalRecipe
from r0.modules.priors.lib.functional_prefix import FunctionalPrefix
from r0.modules.priors.lib.raw_draw_capture import raw_transactions


@dataclass(frozen=True, slots=True)
class SampledFittedFunctionalCausalRecipe(FittedFunctionalCausalRecipe):
    parameter_samples: tuple[tuple[int, SampledFittedTreePlan], ...] = ()
    version: str = "sampled-fitted-functional-causal-v1"

    def __post_init__(self) -> None:
        if (
            self.version != "sampled-fitted-functional-causal-v1"
            or type(self.parameter_samples) is not tuple
            or any(
                type(item) is not tuple
                or len(item) != 2
                or type(item[0]) is not int
                or type(item[1]) is not SampledFittedTreePlan
                for item in self.parameter_samples
            )
            or tuple(i for i, _ in self.parameter_samples) != tuple(f.node for f in self.fits)
        ):
            raise ValueError("sampled fitted graph needs ordered parameter custody for every fit")
        for (_, sampled), fit in zip(self.parameter_samples, self.fits, strict=True):
            if sampled.plan != fit.realization.plan:
                raise ValueError("sampled parameter plan differs from actual fitted backend plan")
            sampled.prior.admit(self.limits)
        for (_, left), (_, right) in zip(
            self.parameter_samples, self.parameter_samples[1:], strict=False
        ):
            if left.final_rng != right.initial_rng:
                raise ValueError("fitted parameter stream chain differs")
        FunctionalPrefix(
            self.nodes, self.task.rows, self.limits, retained=(self.fits, self.parameter_samples)
        ).admit()
        FittedFunctionalCausalRecipe.__post_init__(self)


def sample_parameterized_fitted_graph(
    recipe: FunctionalCausalRecipe,
    priors: tuple[tuple[int, FittedTreeParameterPrior], ...],
    *,
    parameter_rng: np.random.Generator,
    prefix_rng: np.random.Generator,
    input_rng: np.random.Generator,
    target_rng: np.random.Generator,
    backend_rng: np.random.Generator,
) -> SampledFittedFunctionalCausalRecipe:
    if type(recipe) is not FunctionalCausalRecipe or type(priors) is not tuple or not priors:
        raise ValueError("parameterized fitting requires a frozen skeleton and selected priors")
    selected: dict[int, FittedTreeParameterPrior] = {}
    fit_budget = metadata = max_rows = 0
    for item in priors:
        if type(item) is not tuple or len(item) != 2:
            raise ValueError("parameterized fitting requires node/prior pairs")
        index, prior = item
        if (
            type(index) is not int
            or not 0 < index < len(recipe.nodes)
            or index in selected
            or type(prior) is not FittedTreeParameterPrior
        ):
            raise ValueError("invalid or repeated sampled fit selection")
        node = recipe.nodes[index]
        if (
            not node.parents
            or node.pipeline is not None
            or (prior.base.input_width, prior.base.output_width)
            != (node.function.input_width, node.function.output_width)
        ):
            raise ValueError("sampled fit geometry or pre-fitted pipeline differs")
        prior.admit(recipe.limits)
        options = prior.options()
        fit_budget += max(
            p.tree_bound
            * (2 ** (p.depth + 1) - 1)
            * (4 + (p.classes if p.task == "classification" and p.family != "gbdt" else 1))
            + 4 * p.rows * (p.input_width + p.output_width)
            for p in options
        )
        max_rows = max(max_rows, *(p.rows for p in options))
        metadata += 2 * prior.raw_limits.state_bytes + len(prior.effective_laws) * (
            1024 + 2 * prior.raw_limits.state_bytes
        )
        selected[index] = prior
    base = FunctionalPrefix(recipe.nodes, max_rows, recipe.limits).admit()
    recipe.limits.admit_dimensions(
        tuple(n.function.output_width for n in recipe.nodes),
        tuple(n.parents for n in recipe.nodes),
        rows=recipe.task.rows,
        calibration_rows=max_rows,
        features=recipe.task.features,
        outputs=len(recipe.target),
        extra_parameters=base["parameters"] + fit_budget + (metadata + 127) // 128,
    )
    callers = {
        "parameters": parameter_rng,
        "prefix": prefix_rng,
        "input": input_rng,
        "target": target_rng,
        "backend": backend_rng,
    }
    with raw_transactions(callers, None) as (streams, _):
        samples = tuple(
            (i, sample_fitted_tree_plan(p, rng=streams["parameters"], limits=recipe.limits))
            for i, p in sorted(selected.items())
        )
        fitted = sample_fitted_graph(
            recipe,
            tuple((i, s.plan) for i, s in samples),
            prefix_rng=streams["prefix"],
            input_rng=streams["input"],
            target_rng=streams["target"],
            backend_rng=streams["backend"],
        )
        return SampledFittedFunctionalCausalRecipe(
            fitted.task,
            fitted.nodes,
            fitted.observed,
            fitted.target,
            fitted.limits,
            fits=fitted.fits,
            parameter_samples=samples,
        )
