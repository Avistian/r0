"""Sample fitted mechanisms on independent frozen ancestor populations."""

from dataclasses import asdict, dataclass, replace

import numpy as np
import torch

from r0.modules.priors.lib.fitted_tree_fitting import FittedTreeRealization, fit_fitted_tree
from r0.modules.priors.lib.fitted_tree_plan import FittedTreePlan
from r0.modules.priors.lib.functional_causal import FunctionalCausalRecipe
from r0.modules.priors.lib.functional_prefix import FunctionalPrefix
from r0.modules.priors.lib.generator_snapshot import GeneratorSnapshot, snapshot_generator
from r0.modules.priors.lib.joint_point_realization import frozen_matrix
from r0.modules.priors.lib.raw_draw_capture import RawDrawLedger, raw_transactions
from r0.modules.priors.lib.raw_draw_records import RawDrawRecord
from r0.modules.priors.lib.representatives import record_fingerprint


@dataclass(frozen=True, slots=True)
class FittedNodeFit:
    node: int
    prefix_digest: str
    realization: FittedTreeRealization
    prefix_initial: GeneratorSnapshot
    prefix_final: GeneratorSnapshot
    prefix_draws: tuple[RawDrawRecord, ...]

    def __post_init__(self) -> None:
        if (
            type(self.node) is not int
            or self.node < 1
            or type(self.prefix_digest) is not str
            or len(self.prefix_digest) != 64
            or type(self.realization) is not FittedTreeRealization
            or self.realization.input_law != "independent-frozen-ancestor-prefix"
            or type(self.prefix_initial) is not GeneratorSnapshot
            or type(self.prefix_final) is not GeneratorSnapshot
            or type(self.prefix_draws) is not tuple
            or any(type(d) is not RawDrawRecord for d in self.prefix_draws)
        ):
            raise ValueError("fitted node requires independent ancestor custody")


@dataclass(frozen=True, slots=True)
class FittedFunctionalCausalRecipe(FunctionalCausalRecipe):
    fits: tuple[FittedNodeFit, ...] = ()
    version: str = "fitted-functional-causal-v1"

    def __post_init__(self) -> None:
        FunctionalCausalRecipe.__post_init__(self)
        if (
            self.version != "fitted-functional-causal-v1"
            or type(self.fits) is not tuple
            or not self.fits
            or any(type(f) is not FittedNodeFit for f in self.fits)
            or tuple(f.node for f in self.fits) != tuple(sorted({f.node for f in self.fits}))
        ):
            raise ValueError("fitted graph requires ordered unique node fit records")
        for fit in self.fits:
            if (
                fit.node >= len(self.nodes)
                or not self.nodes[fit.node].parents
                or fit.realization.function != self.nodes[fit.node].function
                or fit.prefix_digest
                != record_fingerprint([asdict(n) for n in self.nodes[: fit.node]])
            ):
                raise ValueError("fitted graph ancestor or equation binding differs")
        # Retained input/target/raw tables count against graph metadata caps too.
        FunctionalPrefix(self.nodes, self.task.rows, self.limits, retained=(self.fits,)).admit()
        self.validate_custody()

    def validate_custody(self) -> None:
        from r0.modules.priors.lib.functional_causal_evaluation import sample_functional_prefix

        for fit in self.fits:
            rng = fit.prefix_initial.restore()
            ledger = RawDrawLedger()
            population = sample_functional_prefix(
                FunctionalPrefix(
                    self.nodes[: fit.node],
                    fit.realization.plan.rows,
                    self.limits,
                    retained=(self.fits,),
                ),
                rng=rng,
                owner=("fitted-graph", f"node:{fit.node}"),
                capture=ledger,
            )
            inputs = torch.cat(
                tuple(population.states[p] for p in self.nodes[fit.node].parents), dim=1
            ).numpy()
            if (
                frozen_matrix(inputs) != fit.realization.inputs
                or ledger.freeze() != fit.prefix_draws
                or snapshot_generator(rng) != fit.prefix_final
            ):
                raise ValueError("fitted ancestor population differs from retained prefix custody")


def sample_fitted_graph(
    recipe: FunctionalCausalRecipe,
    plans: tuple[tuple[int, FittedTreePlan], ...],
    *,
    prefix_rng: np.random.Generator,
    input_rng: np.random.Generator,
    target_rng: np.random.Generator,
    backend_rng: np.random.Generator,
) -> FittedFunctionalCausalRecipe:
    """Replace declared nonroot functions; task/query rows never enter the fit path.

    A sampled geometry graph's ``runtime`` is accepted, as is an authored graph.
    The supplied runtime is the explicit frozen skeleton; its selector provenance
    stays with its original construction archive.
    """
    from r0.modules.priors.lib.functional_causal_evaluation import sample_functional_prefix

    if type(recipe) is not FunctionalCausalRecipe or type(plans) is not tuple or not plans:
        raise ValueError("fitted graph requires a frozen uncalibrated skeleton and fit plans")
    selected: dict[int, FittedTreePlan] = {}
    for item in plans:
        if type(item) is not tuple or len(item) != 2:
            raise ValueError("fitted node plans require index/plan pairs")
        index, plan = item
        if (
            type(index) is not int
            or not 0 < index < len(recipe.nodes)
            or index in selected
            or type(plan) is not FittedTreePlan
        ):
            raise ValueError("invalid or duplicate fitted node plan")
        node = recipe.nodes[index]
        if not node.parents or node.pipeline is not None:
            raise ValueError(
                "fitted replacement requires nonroot nodes without pre-fitted pipelines"
            )
        if (plan.input_width, plan.output_width) != (
            node.function.input_width,
            node.function.output_width,
        ):
            raise ValueError("fitted plan geometry differs from selected node")
        plan.admit(recipe.limits)
        selected[index] = plan
    callers = {
        "calibration": prefix_rng,
        "input": input_rng,
        "target": target_rng,
        "backend": backend_rng,
    }
    with raw_transactions(callers, None) as (streams, _):
        nodes = list(recipe.nodes)
        fits: list[FittedNodeFit] = []
        for index, plan in sorted(selected.items()):
            prefix = FunctionalPrefix(
                tuple(nodes[:index]), plan.rows, recipe.limits, retained=(tuple(fits),)
            )
            ledger = RawDrawLedger()
            initial = snapshot_generator(streams["calibration"])
            population = sample_functional_prefix(
                prefix,
                rng=streams["calibration"],
                owner=("fitted-graph", f"node:{index}"),
                capture=ledger,
            )
            inputs = torch.cat(
                tuple(population.states[p] for p in nodes[index].parents), dim=1
            ).numpy()
            fit = fit_fitted_tree(
                plan,
                input_rng=streams["input"],
                target_rng=streams["target"],
                backend_rng=streams["backend"],
                limits=recipe.limits,
                inputs=inputs,
            )
            fits.append(
                FittedNodeFit(
                    index,
                    record_fingerprint([asdict(n) for n in nodes[:index]]),
                    fit,
                    initial,
                    snapshot_generator(streams["calibration"]),
                    ledger.freeze(),
                )
            )
            nodes[index] = replace(nodes[index], function=fit.function)
        return FittedFunctionalCausalRecipe(
            recipe.task,
            tuple(nodes),
            recipe.observed,
            recipe.target,
            recipe.limits,
            fits=tuple(fits),
        )
