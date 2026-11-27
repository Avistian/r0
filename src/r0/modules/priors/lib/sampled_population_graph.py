"""Compose parameter sampling with explicit fitting lifecycles under one transaction."""

from dataclasses import dataclass

import numpy as np

from r0.modules.priors.lib.fitted_population_fit import admit_population_fit
from r0.modules.priors.lib.fitted_population_graph import (
    PopulationFittedGraph,
    fit_population_graph,
)
from r0.modules.priors.lib.fitted_population_plan import MixedFitPopulation, PopulationTreePlan
from r0.modules.priors.lib.functional_causal import FunctionalCausalRecipe
from r0.modules.priors.lib.functional_prefix import FunctionalPrefix
from r0.modules.priors.lib.population_parameter_prior import PopulationTreeParameterPrior
from r0.modules.priors.lib.population_parameter_sampling import (
    SampledPopulationTreePlan,
    sample_population_tree_plan,
)
from r0.modules.priors.lib.raw_draw_capture import raw_transactions


@dataclass(frozen=True, slots=True)
class SampledPopulationFittedGraph(PopulationFittedGraph):
    parameter_samples: tuple[tuple[int, SampledPopulationTreePlan], ...] = ()
    version: str = "sampled-population-fitted-functional-causal-v1"

    def __post_init__(self) -> None:
        if (
            self.version != "sampled-population-fitted-functional-causal-v1"
            or type(self.parameter_samples) is not tuple
            or not self.parameter_samples
            or any(
                type(pair) is not tuple
                or len(pair) != 2
                or type(pair[0]) is not int
                or type(pair[1]) is not SampledPopulationTreePlan
                for pair in self.parameter_samples
            )
            or tuple(i for i, _ in self.parameter_samples) != tuple(f.node for f in self.fits)
        ):
            raise ValueError("sampled population graph needs parameter custody for every fit")
        for (_, sampled), fit in zip(self.parameter_samples, self.fits, strict=True):
            if sampled.plan != fit.realization.plan or sampled.prior.population != self.population:
                raise ValueError("sampled population plan or lifecycle differs from actual fit")
            sampled.prior.admit(self.limits)
        for (_, left), (_, right) in zip(
            self.parameter_samples[:-1], self.parameter_samples[1:], strict=True
        ):
            if left.final_rng != right.initial_rng:
                raise ValueError("population parameter stream chain differs")
        FunctionalPrefix(
            self.nodes, self.task.rows, self.limits, retained=(self.fits, self.parameter_samples)
        ).admit()
        PopulationFittedGraph.__post_init__(self)


def sample_parameterized_population_graph(
    recipe: FunctionalCausalRecipe,
    priors: tuple[tuple[int, PopulationTreeParameterPrior], ...],
    *,
    population: str,
    seed: int,
    task_id: int,
    parameter_rng: np.random.Generator,
    input_rng: np.random.Generator,
    target_rng: np.random.Generator,
    backend_rng: np.random.Generator,
    mixed: MixedFitPopulation | None = None,
) -> SampledPopulationFittedGraph:
    if (
        type(recipe) is not FunctionalCausalRecipe
        or type(priors) is not tuple
        or not priors
        or population not in ("support-only", "batch-refit-reference", "mixed-independent")
        or any(type(v) is not int or not 0 <= v < 2**63 for v in (seed, task_id))
        or (mixed is not None) != (population == "mixed-independent")
        or mixed is not None
        and type(mixed) is not MixedFitPopulation
    ):
        raise ValueError("sampled population fitting requires a skeleton, lifecycle and identities")
    selected: dict[int, PopulationTreeParameterPrior] = {}
    fit_budget = metadata = 0
    max_rows = 1
    for item in priors:
        if type(item) is not tuple or len(item) != 2:
            raise ValueError("population parameter selections require node/prior pairs")
        index, prior = item
        if (
            type(index) is not int
            or not 0 < index < len(recipe.nodes)
            or index in selected
            or type(prior) is not PopulationTreeParameterPrior
            or prior.population != population
        ):
            raise ValueError("invalid population parameter selection or conflicting lifecycle")
        node = recipe.nodes[index]
        if (
            not node.parents
            or node.pipeline is not None
            or (prior.base.input_width, prior.base.output_width)
            != (node.function.input_width, node.function.output_width)
        ):
            raise ValueError("sampled population fit geometry or pipeline differs")
        rows = recipe.task.context_rows if population == "support-only" else recipe.task.rows
        if population != "mixed-independent" and prior.base.rows != rows:
            raise ValueError("support/batch parameter prior must bind the actual episode rows")
        prior.admit(recipe.limits)
        options = prior.options()
        for plan in options:
            assert isinstance(plan, PopulationTreePlan)
            admit_population_fit(plan, recipe.limits)
        fit_budget += max(
            8 * p.rows * (p.input_width + p.output_width)
            + 6000
            + 400 * (p.input_width + p.output_width)
            + 2
            * p.tree_bound
            * (2 ** (p.depth + 1) - 1)
            * (4 + (p.classes if p.task == "classification" and p.family != "gbdt" else 1))
            for p in options
        )
        max_rows = max(max_rows, *(p.rows for p in options))
        metadata += 2 * prior.raw_limits.state_bytes + len(prior.effective_laws) * (
            1024 + 2 * prior.raw_limits.state_bytes
        )
        selected[index] = prior
    recipe.limits.admit_dimensions(
        tuple(n.function.output_width for n in recipe.nodes),
        tuple(n.parents for n in recipe.nodes),
        rows=recipe.task.rows,
        calibration_rows=max_rows,
        features=recipe.task.features,
        outputs=len(recipe.target),
        extra_parameters=recipe.admit()["parameters"] + fit_budget + (metadata + 127) // 128,
    )
    with raw_transactions(
        {
            "parameters": parameter_rng,
            "input": input_rng,
            "target": target_rng,
            "backend": backend_rng,
        },
        None,
    ) as (streams, _):
        samples = tuple(
            (i, sample_population_tree_plan(p, rng=streams["parameters"], limits=recipe.limits))
            for i, p in sorted(selected.items())
        )
        fitted = fit_population_graph(
            recipe,
            tuple((i, s.plan) for i, s in samples),
            population=population,
            seed=seed,
            task_id=task_id,
            mixed=mixed,
            input_rng=streams["input"],
            target_rng=streams["target"],
            backend_rng=streams["backend"],
        )
        return SampledPopulationFittedGraph(
            fitted.task,
            fitted.nodes,
            fitted.observed,
            fitted.target,
            fitted.limits,
            fits=fitted.fits,
            population=population,
            fit_seed=seed,
            fit_task_id=task_id,
            parameter_samples=samples,
        )
