"""Immutable graph populations bind every receipt to its node and original input rows."""

from dataclasses import asdict, dataclass, replace

import numpy as np

from r0.modules.priors.lib.causal_limits import CausalLimits
from r0.modules.priors.lib.exact_frozen_record import exact_frozen_equal
from r0.modules.priors.lib.generator_snapshot import snapshot_generator
from r0.modules.priors.lib.joint_function_admission import function_record_cost
from r0.modules.priors.lib.recursive_forest_sampling import (
    SampledRecursiveForest,
    sample_recursive_forest,
)
from r0.modules.priors.lib.recursive_population_graph_plan import RecursivePopulationGraphRequest
from r0.modules.priors.lib.recursive_population_graph_runtime import (
    PopulationGraphEvaluation,
    evaluate_population_dag,
)
from r0.modules.priors.lib.recursive_population_records import (
    population_identity,
    population_matrix,
)
from r0.modules.priors.lib.representatives import record_fingerprint


@dataclass(frozen=True, slots=True)
class RecursiveNodePopulation:
    node: int
    sample: SampledRecursiveForest
    inputs: tuple[tuple[float, ...], ...]
    version: str = "recursive-node-population-v1"

    def __post_init__(self) -> None:
        if (
            self.version != "recursive-node-population-v1"
            or type(self.node) is not int
            or self.node < 0
            or type(self.sample) is not SampledRecursiveForest
            or self.sample.plan.lifecycle != "fresh-forest-per-row"
            or type(self.inputs) is not tuple
            or len(self.inputs) != self.sample.plan.populations
            or any(
                type(row) is not tuple or len(row) != self.sample.plan.input_width
                for row in self.inputs
            )
            or any(type(v) is not float or not np.isfinite(v) for row in self.inputs for v in row)
        ):
            raise ValueError("node population requires complete ordered finite original input rows")


@dataclass(frozen=True, slots=True)
class RecursivePopulationGraph:
    request: RecursivePopulationGraphRequest
    bindings: tuple[RecursiveNodePopulation, ...]
    seed: int
    task_id: int
    version: str = "recursive-population-graph-v1"

    @property
    def task(self):
        return self.request.task

    @property
    def limits(self):
        return self.request.limits

    def __post_init__(self) -> None:
        if (
            self.version != "recursive-population-graph-v1"
            or type(self.request) is not RecursivePopulationGraphRequest
        ):
            raise ValueError("population DAG requires its distinct request and version")
        population_identity(self.seed, self.task_id)
        self.validate_replay()

    def replay(self, *, limits: CausalLimits | None = None) -> PopulationGraphEvaluation:
        from r0.modules.priors.lib.recursive_forest_plan import recursive_effective_limits

        effective = recursive_effective_limits(self.limits, limits or self.limits)
        numbers, storage = function_record_cost(self, effective)
        if numbers >= effective.parameters or 8 * storage >= effective.working_bytes:
            raise ValueError("complete population DAG metadata exceeds limits before replay")
        self.request.admit(
            limits=replace(
                effective,
                parameters=effective.parameters - numbers,
                working_bytes=effective.working_bytes - 8 * storage,
            )
        )
        if (
            type(self.bindings) is not tuple
            or any(type(b) is not RecursiveNodePopulation for b in self.bindings)
            or tuple(b.node for b in self.bindings)
            != tuple(p.node for p in self.request.replacements)
        ):
            raise ValueError("population DAG binding order differs from declared replacements")
        for i, (binding, plan) in enumerate(
            zip(self.bindings, self.request.replacements, strict=True)
        ):
            if (
                not exact_frozen_equal(binding.sample.plan, plan.forest)
                or not exact_frozen_equal(binding.sample.limits, self.request.limits)
                or not exact_frozen_equal(binding.sample.raw_limits, self.request.raw_limits)
                or i > 0
                and not exact_frozen_equal(
                    self.bindings[i - 1].sample.final, binding.sample.initial
                )
            ):
                raise ValueError(
                    "population DAG receipt differs from its plan or owned construction stream"
                )
            binding.sample.validate_replay(limits=limits or self.limits)
        evaluation = evaluate_population_dag(
            self.request,
            {b.node: b.sample for b in self.bindings},
            seed=self.seed,
            task_id=self.task_id,
        )
        for binding in self.bindings:
            if not exact_frozen_equal(
                population_matrix(evaluation.context.inputs[binding.node]), binding.inputs
            ):
                raise ValueError("population DAG input rows differ from node/row ownership")
        return evaluation

    def validate_replay(self, *, limits: CausalLimits | None = None) -> None:
        self.replay(limits=limits)


def realize_recursive_population_graph(
    request: RecursivePopulationGraphRequest,
    *,
    seed: int,
    task_id: int,
    rng: np.random.Generator | None = None,
) -> RecursivePopulationGraph:
    if (
        type(request) is not RecursivePopulationGraphRequest
        or rng is not None
        and type(rng) is not np.random.Generator
    ):
        raise ValueError("population DAG realization requires its request and owned NumPy stream")
    population_identity(seed, task_id)
    request.admit()
    caller = (
        rng
        if rng is not None
        else np.random.default_rng(
            int(
                record_fingerprint(
                    ("recursive-population-dag-construction-v1", asdict(request), seed, task_id)
                ),
                16,
            )
        )
    )
    initial = snapshot_generator(caller, byte_limit=request.raw_limits.state_bytes)
    owned = initial.restore(byte_limit=request.raw_limits.state_bytes)
    samples = {
        p.node: sample_recursive_forest(
            p.forest, rng=owned, limits=request.limits, raw_limits=request.raw_limits
        )
        for p in request.replacements
    }
    evaluation = evaluate_population_dag(request, samples, seed=seed, task_id=task_id)
    bindings = tuple(
        RecursiveNodePopulation(
            p.node, samples[p.node], population_matrix(evaluation.context.inputs[p.node])
        )
        for p in request.replacements
    )
    result = RecursivePopulationGraph(request, bindings, seed, task_id)
    if snapshot_generator(caller, byte_limit=request.raw_limits.state_bytes) != initial:
        raise ValueError("caller population DAG construction stream changed during realization")
    caller.bit_generator.state = owned.bit_generator.state
    return result
