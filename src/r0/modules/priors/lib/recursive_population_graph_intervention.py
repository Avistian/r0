"""Matched and fresh graph populations retain baseline receipts and explicit runtime cuts."""

from dataclasses import asdict, dataclass

import numpy as np

from r0.modules.priors.lib.causal_intervention_records import CausalInterventionRecipe
from r0.modules.priors.lib.causal_limits import CausalLimits
from r0.modules.priors.lib.exact_frozen_record import exact_frozen_equal
from r0.modules.priors.lib.generator_snapshot import snapshot_generator
from r0.modules.priors.lib.joint_function_admission import function_record_cost
from r0.modules.priors.lib.recursive_forest_sampling import (
    SampledRecursiveForest,
    sample_recursive_forest,
)
from r0.modules.priors.lib.recursive_population_graph_output import population_graph_task
from r0.modules.priors.lib.recursive_population_graph_records import RecursivePopulationGraph
from r0.modules.priors.lib.recursive_population_graph_runtime import (
    PopulationGraphEvaluation,
    evaluate_population_dag,
)
from r0.modules.priors.lib.recursive_population_records import (
    population_identity,
    population_matrix,
)
from r0.modules.priors.lib.representatives import record_fingerprint
from r0.modules.priors.lib.training_task import TrainingTask


def population_graph_stream(base: RecursivePopulationGraph, population: int) -> np.random.Generator:
    engine = base.bindings[0].sample.initial.engine
    engines = {
        name: getattr(np.random, name)
        for name in ("PCG64", "PCG64DXSM", "Philox", "SFC64", "MT19937")
    }
    seed = int(
        record_fingerprint(
            (
                "recursive-population-dag-intervention-v1",
                asdict(base.bindings[0].sample.initial),
                base.seed,
                base.task_id,
                population,
            )
        ),
        16,
    )
    return np.random.Generator(engines[engine](seed))


def admit_population_graph_intervention(
    base: RecursivePopulationGraph,
    change: CausalInterventionRecipe,
    limits: CausalLimits | None = None,
) -> None:
    from r0.modules.priors.lib.recursive_forest_plan import recursive_effective_limits

    effective = recursive_effective_limits(base.limits, limits or base.limits)
    report = base.request.admit(limits=effective)
    after = change.admit()
    numbers, storage = function_record_cost((base, change), effective)
    if report["parameters"] * 2 + after["parameters"] + numbers > effective.parameters or (
        2 * report["estimated_working_bytes"] + after["estimated_working_bytes"] + 8 * storage
        > effective.working_bytes
    ):
        raise ValueError("population DAG intervention aggregate custody exceeds limits")


@dataclass(frozen=True, slots=True)
class RecursivePopulationGraphIntervention:
    base: RecursivePopulationGraph
    change: CausalInterventionRecipe
    samples: tuple[SampledRecursiveForest, ...]
    inputs: tuple[tuple[int, tuple[tuple[float, ...], ...] | None], ...]
    version: str = "recursive-population-graph-intervention-v1"

    @property
    def task(self):
        return self.base.task

    @property
    def limits(self):
        return self.base.limits

    def __post_init__(self) -> None:
        if (
            self.version != "recursive-population-graph-intervention-v1"
            or type(self.base) is not RecursivePopulationGraph
            or type(self.change) is not CausalInterventionRecipe
            or not exact_frozen_equal(self.change.base, self.base.request.base)
        ):
            raise ValueError("population DAG intervention must retain the exact original base")
        self.validate_replay()

    def replay(self, *, limits: CausalLimits | None = None) -> PopulationGraphEvaluation:
        admit_population_graph_intervention(self.base, self.change, limits)
        function_record_cost(self, limits or self.limits)
        self.base.validate_replay(limits=limits)
        if (
            type(self.samples) is not tuple
            or len(self.samples) != len(self.base.bindings)
            or any(type(sample) is not SampledRecursiveForest for sample in self.samples)
        ):
            raise ValueError("population DAG intervention requires every ordered receipt")
        if self.change.mode == "matched":
            if not exact_frozen_equal(self.samples, tuple(b.sample for b in self.base.bindings)):
                raise ValueError("matched population DAG intervention cannot change any forest")
        else:
            initial = snapshot_generator(
                population_graph_stream(self.base, self.change.population),
                byte_limit=self.base.request.raw_limits.state_bytes,
            )
            if not exact_frozen_equal(initial, self.samples[0].initial):
                raise ValueError(
                    "fresh population DAG forest stream differs from population identity"
                )
        for i, (sample, binding) in enumerate(zip(self.samples, self.base.bindings, strict=True)):
            if (
                not exact_frozen_equal(sample.plan, binding.sample.plan)
                or not exact_frozen_equal(sample.limits, binding.sample.limits)
                or not exact_frozen_equal(sample.raw_limits, binding.sample.raw_limits)
                or i > 0
                and not exact_frozen_equal(self.samples[i - 1].final, sample.initial)
            ):
                raise ValueError("population DAG intervention receipt binding differs")
            sample.validate_replay(limits=limits or self.limits)
        result = evaluate_population_dag(
            self.base.request,
            {b.node: s for b, s in zip(self.base.bindings, self.samples, strict=True)},
            seed=self.base.seed,
            task_id=self.base.task_id,
            intervention=self.change,
        )
        expected = tuple(
            (
                b.node,
                population_matrix(result.context.inputs[b.node])
                if b.node in result.context.inputs
                else None,
            )
            for b in self.base.bindings
        )
        if not exact_frozen_equal(expected, self.inputs):
            raise ValueError(
                "population DAG intervention original inputs or bypass ownership differ"
            )
        return result

    def validate_replay(self, *, limits: CausalLimits | None = None) -> None:
        self.replay(limits=limits)


def intervene_recursive_population_graph(
    base: RecursivePopulationGraph, change: CausalInterventionRecipe
) -> RecursivePopulationGraphIntervention:
    if (
        type(base) is not RecursivePopulationGraph
        or type(change) is not CausalInterventionRecipe
        or not exact_frozen_equal(change.base, base.request.base)
    ):
        raise ValueError("population DAG intervention requires its exact original base")
    admit_population_graph_intervention(base, change)
    if change.mode == "matched":
        samples = tuple(b.sample for b in base.bindings)
    else:
        stream = population_graph_stream(base, change.population)
        samples = tuple(
            sample_recursive_forest(
                p.forest, rng=stream, limits=base.limits, raw_limits=base.request.raw_limits
            )
            for p in base.request.replacements
        )
    result = evaluate_population_dag(
        base.request,
        {b.node: s for b, s in zip(base.bindings, samples, strict=True)},
        seed=base.seed,
        task_id=base.task_id,
        intervention=change,
    )
    inputs = tuple(
        (
            b.node,
            population_matrix(result.context.inputs[b.node])
            if b.node in result.context.inputs
            else None,
        )
        for b in base.bindings
    )
    return RecursivePopulationGraphIntervention(base, change, samples, inputs)


def generate_population_graph_intervention(
    recipe: RecursivePopulationGraphIntervention, *, seed: int, task_id: int, device: str = "cpu"
) -> TrainingTask:
    population_identity(seed, task_id)
    if (
        type(recipe) is not RecursivePopulationGraphIntervention
        or device != "cpu"
        or (seed, task_id) != (recipe.base.seed, recipe.base.task_id)
    ):
        raise ValueError("population DAG intervention retains original identity and CPU execution")
    evaluation = recipe.replay()
    metadata = {
        "change": asdict(recipe.change),
        "mode": recipe.change.mode,
        "population": recipe.change.population,
        "disabled_population_nodes": sorted(evaluation.context.disabled),
        "bypassed_nodes": [
            i
            for i in range(len(recipe.base.request.base.nodes))
            if i not in evaluation.context.inputs
        ],
        "after_receipts": [
            {
                "node": b.node,
                "initial": s.initial.digest,
                "final": s.final.digest,
                "inputs_digest": record_fingerprint(dict(recipe.inputs)[b.node]),
            }
            for b, s in zip(recipe.base.bindings, recipe.samples, strict=True)
        ],
        "scope": (
            "baseline custody replay precedes the separately executed "
            "counterfactual; full cuts bypass runtime mechanisms"
        ),
    }
    return population_graph_task(
        recipe.base, evaluation, config=asdict(recipe), intervention=metadata
    )


def generate_population_graph_pair(
    base: RecursivePopulationGraph, change: CausalInterventionRecipe
):
    """Compare explicit causal changes without refitting or rewriting the baseline archive."""
    from r0.modules.priors.lib.pairs import TaskPair
    from r0.modules.priors.lib.recursive_population_graph_output import (
        generate_recursive_population_graph,
    )
    from r0.modules.priors.lib.representatives import encode

    changed = intervene_recursive_population_graph(base, change)
    before = generate_recursive_population_graph(base, seed=base.seed, task_id=base.task_id)
    after = generate_population_graph_intervention(changed, seed=base.seed, task_id=base.task_id)
    from r0.modules.priors.lib.population_graph_comparison import population_graph_pair_card

    return TaskPair(before, after, encode(population_graph_pair_card(base, changed, before, after)))
