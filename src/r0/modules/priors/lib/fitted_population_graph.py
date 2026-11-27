"""Explicit support-only, batch-reference and mixed independent structural fitting."""

from dataclasses import asdict, dataclass, replace

import numpy as np
import torch

from r0.modules.priors.lib.fitted_population_fit import (
    PopulationTreeFit,
    admit_population_fit,
    fit_population_tree,
)
from r0.modules.priors.lib.fitted_population_plan import MixedFitPopulation, PopulationTreePlan
from r0.modules.priors.lib.functional_causal import FunctionalCausalRecipe
from r0.modules.priors.lib.functional_prefix import FunctionalPrefix
from r0.modules.priors.lib.joint_point_realization import frozen_matrix
from r0.modules.priors.lib.raw_draw_capture import raw_transactions
from r0.modules.priors.lib.representatives import record_fingerprint


@dataclass(frozen=True, slots=True)
class SupportPopulationFunctionalRecipe(FunctionalCausalRecipe):
    """Authored equations retaining the support/query draw contract after editing."""

    population: str = "support-only"
    version: str = "support-population-functional-causal-v1"

    def __post_init__(self) -> None:
        if (
            self.population != "support-only"
            or self.version != "support-population-functional-causal-v1"
        ):
            raise ValueError("authored support graph requires its sampling contract")
        FunctionalCausalRecipe.__post_init__(self)


@dataclass(frozen=True, slots=True)
class PopulationNodeFit:
    node: int
    prefix_digest: str
    realization: PopulationTreeFit

    def __post_init__(self) -> None:
        if (
            type(self.node) is not int
            or self.node < 1
            or type(self.prefix_digest) is not str
            or len(self.prefix_digest) != 64
            or type(self.realization) is not PopulationTreeFit
        ):
            raise ValueError("population node fit requires its equation and prefix binding")


@dataclass(frozen=True, slots=True)
class PopulationFittedGraph(FunctionalCausalRecipe):
    fits: tuple[PopulationNodeFit, ...] = ()
    population: str = "support-only"
    fit_seed: int = 0
    fit_task_id: int = 0
    version: str = "population-fitted-functional-causal-v1"
    lifecycle: str = "explicit-population-freeze-r0-v1"

    def __post_init__(self) -> None:
        if (
            self.version != "population-fitted-functional-causal-v1"
            or self.lifecycle != "explicit-population-freeze-r0-v1"
            or self.population not in ("support-only", "batch-refit-reference", "mixed-independent")
            or any(
                type(v) is not int or not 0 <= v < 2**63 for v in (self.fit_seed, self.fit_task_id)
            )
            or type(self.fits) is not tuple
            or not self.fits
            or any(type(f) is not PopulationNodeFit for f in self.fits)
            or tuple(f.node for f in self.fits) != tuple(sorted({f.node for f in self.fits}))
        ):
            raise ValueError(
                "population graph requires explicit lifecycle and ordered fit bindings"
            )
        FunctionalCausalRecipe.__post_init__(self)
        FunctionalPrefix(self.nodes, self.task.rows, self.limits, retained=(self.fits,)).admit()
        for earlier, later in zip(self.fits[:-1], self.fits[1:], strict=True):
            if earlier.realization.final_rng != later.realization.initial_rng:
                raise ValueError("population fit streams must continue across ordered nodes")
        for fit in self.fits:
            if (
                fit.node >= len(self.nodes)
                or not self.nodes[fit.node].parents
                or fit.realization.population != self.population
                or fit.realization.function != self.nodes[fit.node].function
                or fit.prefix_digest
                != record_fingerprint([asdict(n) for n in self.nodes[: fit.node]])
            ):
                raise ValueError("population fit equation, lifecycle or ancestor binding differs")
            if self.population != "mixed-independent":
                rows = (
                    self.task.context_rows if self.population == "support-only" else self.task.rows
                )
                if fit.realization.plan.rows != rows:
                    raise ValueError(
                        "fitted population row count differs from declared episode split"
                    )
                inputs = population_parent_inputs(self, fit.node, self.fit_seed, self.fit_task_id)
                if frozen_matrix(inputs) != fit.realization.inputs:
                    raise ValueError("fitted population differs from original episode parent rows")


def population_parent_inputs(
    recipe: FunctionalCausalRecipe,
    node: int,
    seed: int,
    task_id: int,
    *,
    population: str | None = None,
) -> np.ndarray:
    from r0.modules.priors.lib.functional_causal_evaluation import (
        evaluate_functional_prefix,
        functional_rng,
    )

    mode = population or getattr(recipe, "population", None)
    rows = recipe.task.context_rows if mode == "support-only" else recipe.task.rows
    prefix = FunctionalPrefix(recipe.nodes[:node], rows, recipe.limits)

    def rng(*keys: object) -> np.random.Generator:
        return (
            functional_rng(seed, task_id, "fit-population-v1", "support", *keys)
            if mode == "support-only"
            else functional_rng(seed, task_id, *keys)
        )

    values = evaluate_functional_prefix(prefix, rng=rng)
    return torch.cat(tuple(values.states[p] for p in recipe.nodes[node].parents), dim=1).numpy()


def fit_population_graph(
    recipe: FunctionalCausalRecipe,
    plans: tuple[tuple[int, PopulationTreePlan], ...],
    *,
    population: str,
    seed: int,
    task_id: int,
    input_rng: np.random.Generator,
    target_rng: np.random.Generator,
    backend_rng: np.random.Generator,
    mixed: MixedFitPopulation | None = None,
) -> PopulationFittedGraph:
    """An explicit fit event creates a frozen function; replay never invokes a backend."""
    if (
        type(recipe) is not FunctionalCausalRecipe
        or type(plans) is not tuple
        or not plans
        or population not in ("support-only", "batch-refit-reference", "mixed-independent")
        or any(type(v) is not int or not 0 <= v < 2**63 for v in (seed, task_id))
        or (mixed is not None) != (population == "mixed-independent")
    ):
        raise ValueError("population graph requires a skeleton, explicit lifecycle and identities")
    selected: dict[int, PopulationTreePlan] = {}
    for pair in plans:
        if type(pair) is not tuple or len(pair) != 2:
            raise ValueError("population plans require node/plan pairs")
        index, plan = pair
        if (
            type(index) is not int
            or not 0 < index < len(recipe.nodes)
            or index in selected
            or type(plan) is not PopulationTreePlan
        ):
            raise ValueError("invalid population node or plan")
        node = recipe.nodes[index]
        if (
            not node.parents
            or node.pipeline is not None
            or (plan.input_width, plan.output_width)
            != (node.function.input_width, node.function.output_width)
        ):
            raise ValueError("population fitting geometry or frozen pipeline differs")
        rows = recipe.task.context_rows if population == "support-only" else recipe.task.rows
        if population != "mixed-independent" and plan.rows != rows:
            raise ValueError("plan rows must equal the actual selected fitting population")
        admit_population_fit(plan, recipe.limits)
        selected[index] = plan
    cells = sum(p.rows * (p.input_width + p.output_width) for p in selected.values())
    recipe.limits.admit_dimensions(
        tuple(n.function.output_width for n in recipe.nodes),
        tuple(n.parents for n in recipe.nodes),
        rows=recipe.task.rows,
        calibration_rows=max(p.rows for p in selected.values()),
        features=recipe.task.features,
        outputs=len(recipe.target),
        extra_parameters=recipe.admit()["parameters"]
        + 8 * cells
        + 6000 * len(selected)
        + sum(
            400 * (p.input_width + p.output_width)
            + 2
            * p.tree_bound
            * (2 ** (p.depth + 1) - 1)
            * (4 + (p.classes if p.task == "classification" and p.family != "gbdt" else 1))
            for p in selected.values()
        ),
    )
    with raw_transactions(
        {"input": input_rng, "target": target_rng, "backend": backend_rng}, None
    ) as (streams, _):
        nodes = list(recipe.nodes)
        fits: list[PopulationNodeFit] = []
        for index, plan in sorted(selected.items()):
            current = replace(recipe, nodes=tuple(nodes))
            inputs = (
                None
                if mixed
                else population_parent_inputs(current, index, seed, task_id, population=population)
            )
            fit = fit_population_tree(
                plan,
                population=population,
                inputs=inputs,
                mixed=mixed,
                limits=recipe.limits,
                input_rng=streams["input"],
                target_rng=streams["target"],
                backend_rng=streams["backend"],
            )
            fits.append(
                PopulationNodeFit(
                    index, record_fingerprint([asdict(n) for n in nodes[:index]]), fit
                )
            )
            nodes[index] = replace(nodes[index], function=fit.function)
        return PopulationFittedGraph(
            recipe.task,
            tuple(nodes),
            recipe.observed,
            recipe.target,
            recipe.limits,
            fits=tuple(fits),
            population=population,
            fit_seed=seed,
            fit_task_id=task_id,
        )
