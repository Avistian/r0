"""Retained finite GP root inputs for authored mixed-mechanism DAGs."""

import json
from dataclasses import asdict, dataclass, replace

import torch

from r0.modules.priors.lib.causal_limits import CausalLimits
from r0.modules.priors.lib.converted_causal_records import ConvertedCausalRecipe
from r0.modules.priors.lib.functional_causal import FunctionalCausalRecipe
from r0.modules.priors.lib.functional_causal_node import FunctionalCausalNode
from r0.modules.priors.lib.gp_column_population import GPColumnPopulation
from r0.modules.priors.lib.joint_function_admission import function_record_cost
from r0.modules.priors.lib.native_causal import NativeCausalRecipe
from r0.modules.priors.lib.native_vector_records import NativeVectorRecipe
from r0.modules.priors.lib.representatives import encode, record_fingerprint
from r0.modules.priors.lib.root_populations import RootPopulations
from r0.modules.priors.lib.training_task import TrainingTask

type GPRootRuntime = (
    FunctionalCausalRecipe | NativeCausalRecipe | NativeVectorRecipe | ConvertedCausalRecipe
)


@dataclass(frozen=True, slots=True)
class GPRootBinding:
    node: int
    columns: tuple[int, ...]
    version: str = "gp-root-binding-v1"

    def __post_init__(self) -> None:
        if (
            type(self.node) is not int
            or self.node < 0
            or type(self.columns) is not tuple
            or not self.columns
            or any(type(i) is not int or i < 0 for i in self.columns)
            or self.version != "gp-root-binding-v1"
        ):
            raise ValueError("GP root binding needs an explicit node and column indices")


@dataclass(frozen=True, slots=True)
class GPRootDAGRecipe:
    source: GPColumnPopulation
    base: GPRootRuntime
    bindings: tuple[GPRootBinding, ...]
    limits: CausalLimits = CausalLimits()
    version: str = "gp-root-dag-v1"

    @property
    def task(self):
        return self.base.task

    def __post_init__(self) -> None:
        if (
            type(self.source) is not GPColumnPopulation
            or type(self.base)
            not in (
                FunctionalCausalRecipe,
                NativeCausalRecipe,
                NativeVectorRecipe,
                ConvertedCausalRecipe,
            )
            or type(self.limits) is not CausalLimits
            or self.version != "gp-root-dag-v1"
            or type(self.bindings) is not tuple
            or not self.bindings
            or any(type(b) is not GPRootBinding for b in self.bindings)
        ):
            raise ValueError("GP root DAG requires complete retained source and authored graph")
        if self.task.rows != self.source.request.rows:
            raise ValueError("GP root DAG must preserve complete original population rows")
        if len({b.node for b in self.bindings}) != len(self.bindings):
            raise ValueError("GP root nodes require unique bindings")
        for binding in self.bindings:
            if binding.node >= len(self.base.nodes):
                raise ValueError("GP root binding node is outside graph")
            node = self.base.nodes[binding.node]
            equation = node if isinstance(node, FunctionalCausalNode) else node.equation
            if equation.parents or equation.root is None:
                raise ValueError("GP population bindings require actual root nodes")
            if len(binding.columns) != equation.function.input_width or any(
                i >= self.source.request.columns for i in binding.columns
            ):
                raise ValueError("GP root binding column geometry differs")
        self.admit()

    def admit(self) -> None:
        for cap in (self.limits, self.source.request.limits, self.base.limits):
            source = self.source.request.admit(cap)
            graph = replace(self.base, limits=cap).admit()
            count, storage = function_record_cost(self, cap)
            if (
                count + source["parameters"] + graph["parameters"] > cap.parameters
                or 4 * storage
                + 4 * (source["estimated_working_bytes"] + graph["estimated_working_bytes"])
                > cap.working_bytes
            ):
                raise ValueError("GP root DAG aggregate source/graph metadata exceeds limits")

    def context(self) -> RootPopulations:
        return RootPopulations(
            tuple(
                (b.node, tuple(tuple(row[j] for j in b.columns) for row in self.source.draw.values))
                for b in self.bindings
            ),
            self.source.population_id,
            self.source.row_ids,
            tuple((b.node, b.columns) for b in self.bindings),
        )


def generate_gp_root_dag(
    recipe: GPRootDAGRecipe, *, seed: int, task_id: int, device: str = "cpu"
) -> TrainingTask:
    if any(type(v) is not int for v in (seed, task_id)) or (seed, task_id) != (
        recipe.source.seed,
        recipe.source.task_id,
    ):
        raise ValueError("GP root DAG requires exact retained population identity")
    if device not in ("cpu", "cuda") or (device == "cuda" and not torch.cuda.is_available()):
        raise ValueError("GP root DAG device is unavailable; no implicit fallback")
    recipe.admit()
    from r0.modules.priors.lib.gp_root_decoding import gp_root_dag_from_dict

    valid = gp_root_dag_from_dict(asdict(recipe), limits=recipe.limits)
    source = valid.source
    context = valid.context()
    if isinstance(valid.base, FunctionalCausalRecipe):
        from r0.modules.priors.lib.functional_causal_evaluation import generate_functional_causal

        task = generate_functional_causal(
            valid.base, seed=seed, task_id=task_id, device=device, root_populations=context
        )
    elif isinstance(valid.base, NativeCausalRecipe):
        from r0.modules.priors.lib.native_evaluation import generate_native_causal

        task = generate_native_causal(
            valid.base, seed=seed, task_id=task_id, device=device, root_populations=context
        )
    elif isinstance(valid.base, NativeVectorRecipe):
        from r0.modules.priors.lib.native_vector_output import generate_native_vector

        task = generate_native_vector(
            valid.base, seed=seed, task_id=task_id, device=device, root_populations=context
        )
    else:
        from r0.modules.priors.lib.converted_output import generate_converted

        task = generate_converted(
            valid.base, seed=seed, task_id=task_id, device=device, root_populations=context
        )
    config = asdict(valid)
    record = json.loads(task.recipe_json)
    record["recipe"] = config
    record["execution"]["implementation"] = valid.version
    card = {
        "version": valid.version,
        "family": "gp_root_dag",
        "population_id": source.population_id,
        "bindings": [asdict(b) for b in valid.bindings],
        "source_target_role": "retained source provenance only; DAG readout defines target",
        "runtime": json.loads(task.mechanism_json),
    }
    source_id = record_fingerprint((config, seed, task_id))
    return replace(
        task,
        source_task_id=source_id,
        episode_id=record_fingerprint((source_id, valid.task.split, valid.task.context_rows)),
        recipe_json=encode(record),
        mechanism_json=encode(card),
    )
