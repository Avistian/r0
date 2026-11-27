"""Replayable causal comparisons, separate from mechanism editing and task adapters."""

import json
from dataclasses import asdict, replace
from typing import Any, cast

import torch

from r0.modules.priors.lib.causal_intervention_execution import InterventionExecution
from r0.modules.priors.lib.causal_intervention_records import (
    CausalBase,
    CausalInterventionRecipe,
    HardIntervention,
    SoftIntervention,
    causal_runtime,
)
from r0.modules.priors.lib.causal_limits import CausalLimits
from r0.modules.priors.lib.functional_causal import FunctionalCausalRecipe
from r0.modules.priors.lib.functional_causal_evaluation import (
    generate_functional_causal,
)
from r0.modules.priors.lib.native_causal import NativeCausalRecipe
from r0.modules.priors.lib.native_evaluation import evaluate_native_graph, generate_native_causal
from r0.modules.priors.lib.native_vector_evaluation import evaluate_native_vector_graph
from r0.modules.priors.lib.native_vector_intervention import (
    NativeVectorDomainIntervention,
    NativeVectorHardIntervention,
)
from r0.modules.priors.lib.native_vector_output import generate_native_vector
from r0.modules.priors.lib.native_vector_records import NativeVectorRecipe
from r0.modules.priors.lib.representatives import encode, record_fingerprint
from r0.modules.priors.lib.sampled_native_graph import SampledNativeGraph
from r0.modules.priors.lib.sampled_native_vector_graph import SampledNativeVectorGraph
from r0.modules.priors.lib.stage_intervention import StageIntervention
from r0.modules.priors.lib.training_task import TrainingTask


def generate_intervened(
    recipe: CausalInterventionRecipe,
    *,
    seed: int,
    task_id: int,
    device: str = "cpu",
    execution: InterventionExecution | None = None,
) -> TrainingTask:
    if device != "cpu":
        raise ValueError("causal intervention execution currently supports explicit CPU only")
    execution = execution or InterventionExecution(recipe)
    original = causal_runtime(recipe.base)
    if not isinstance(
        recipe.base, (FunctionalCausalRecipe, NativeCausalRecipe, NativeVectorRecipe)
    ):
        if isinstance(recipe.base, (SampledNativeGraph, SampledNativeVectorGraph)):
            recipe.base.validate_replay()
        else:
            recipe.base.validate()
    recipe.admit()
    runtime = recipe.effective_runtime()
    if isinstance(runtime, NativeVectorRecipe):
        task = generate_native_vector(
            runtime, seed=seed, task_id=task_id, device=device, execution=execution
        )
    elif isinstance(runtime, NativeCausalRecipe):
        task = generate_native_causal(
            runtime, seed=seed, task_id=task_id, device=device, execution=execution
        )
    else:
        assert isinstance(original, FunctionalCausalRecipe)
        task = generate_functional_causal(
            runtime,
            seed=seed,
            task_id=task_id,
            device=device,
            execution=execution,
            calibration_recipe=original,
        )
    config = asdict(recipe)
    source = record_fingerprint((config, seed, task_id))
    record = json.loads(task.recipe_json)
    record["recipe"] = config
    record["execution"]["implementation"] = recipe.version
    card = json.loads(task.mechanism_json)
    card["intervention"] = execution.metadata()
    card["base_mechanism_identity"] = record_fingerprint(asdict(recipe.base))
    card["ancestry_scope"] = (
        "original graph certificates are conservative; "
        "intervention cuts are explicit in intervention records"
    )
    return replace(
        task,
        source_task_id=source,
        episode_id=record_fingerprint((source, recipe.task.split, recipe.task.context_rows)),
        recipe_json=encode(record),
        mechanism_json=encode(card),
    )


def trace_original(
    recipe: CausalInterventionRecipe, *, seed: int, task_id: int
) -> InterventionExecution:
    trace = InterventionExecution(recipe, baseline=True)
    runtime = causal_runtime(recipe.base)
    if isinstance(runtime, NativeVectorRecipe):
        evaluate_native_vector_graph(runtime, seed=seed, task_id=task_id, execution=trace)
    elif isinstance(runtime, NativeCausalRecipe):
        evaluate_native_graph(runtime, seed=seed, task_id=task_id, execution=trace)
    else:
        generate_functional_causal(
            runtime, seed=seed, task_id=task_id, execution=trace, calibration_recipe=runtime
        )
    return trace


def generate_causal_pair(
    base: CausalBase,
    *,
    interventions: tuple[
        HardIntervention
        | SoftIntervention
        | StageIntervention
        | NativeVectorHardIntervention
        | NativeVectorDomainIntervention,
        ...,
    ],
    mode: str = "matched",
    population: int = 1,
    seed: int,
    task_id: int,
    device: str = "cpu",
):
    from r0.modules.priors.lib.generate import generate_task
    from r0.modules.priors.lib.pairs import TaskPair

    if device != "cpu":
        raise ValueError("causal intervention comparisons currently support explicit CPU only")
    recipe = CausalInterventionRecipe(base, interventions, mode, population)
    before = generate_task(base, seed=seed, task_id=task_id, device=device)
    after = generate_intervened(recipe, seed=seed, task_id=task_id, device=device)
    return TaskPair(
        before,
        after,
        encode(compare_causal_pair(before, after, limits=causal_runtime(base).limits)),
    )


def compare_causal_pair(
    before: TrainingTask, after: TrainingTask, *, limits: CausalLimits | None = None
) -> dict[str, Any]:
    from r0.modules.priors.lib.generate import generate_task, recipe_from_dict

    b, a = json.loads(before.recipe_json), json.loads(after.recipe_json)
    recipe = recipe_from_dict(a["recipe"], limits=limits)
    if not isinstance(recipe, CausalInterventionRecipe) or b["recipe"] != asdict_json(recipe.base):
        raise ValueError("causal comparison must retain its original base recipe")
    if (b["seed"], b["task_id"]) != (a["seed"], a["task_id"]):
        raise ValueError("causal comparison must retain original source row identities")
    seed, task_id = a["seed"], a["task_id"]
    expected_before = generate_task(recipe.base, seed=seed, task_id=task_id)
    execution = InterventionExecution(recipe)
    expected_after = generate_intervened(recipe, seed=seed, task_id=task_id, execution=execution)
    for actual, expected in ((before, expected_before), (after, expected_after)):
        for field in (
            "source_task_id",
            "episode_id",
            "task_type",
            "classes",
            "feature_schema",
            "target_origin",
            "mechanism_json",
        ):
            if getattr(actual, field) != getattr(expected, field):
                raise ValueError("causal comparison metadata disagrees with replay")
        actual_execution = json.loads(actual.recipe_json).get("execution")
        expected_execution = json.loads(expected.recipe_json)["execution"]
        if type(actual_execution) is not dict or any(
            cast(dict[str, Any], actual_execution).get(field) != expected_execution[field]
            for field in ("implementation", "device")
        ):
            raise ValueError("causal comparison execution metadata disagrees with replay")
        for field in ("inputs", "targets", "scores", "context_indices", "query_indices"):
            if not torch.equal(getattr(actual, field).cpu(), getattr(expected, field).cpu()):
                raise ValueError("causal comparison arrays disagree with retained replay")
        if (
            (actual.probabilities is None) != (expected.probabilities is None)
            or actual.probabilities is not None
            and expected.probabilities is not None
            and not torch.equal(actual.probabilities.cpu(), expected.probabilities.cpu())
        ):
            raise ValueError("causal comparison probabilities disagree with replay")
    original = trace_original(recipe, seed=seed, task_id=task_id)
    unchanged_laws = {
        i
        for i in original.traces
        if not any(
            isinstance(c, SoftIntervention) and c.node == i and c.exogenous is not None
            for c in recipe.interventions
        )
    }
    checks: dict[str, bool] = {}
    for i in unchanged_laws:
        left, right = original.traces[i], execution.traces[i]
        for field in (
            "root",
            "exogenous",
            "uniform",
            *(k for k in left if k.startswith("uniform:")),
        ):
            changed_root = field == "root" and any(
                isinstance(c, StageIntervention) and c.node == i and c.root is not None
                for c in recipe.interventions
            )
            if field in left and field in right and not changed_root:
                checks[f"{i}:{field}"] = torch.equal(left[field], right[field])
    stream_checks = {
        key: value == execution.stream_initial[key]
        for key, value in original.stream_initial.items()
        if key in execution.stream_initial
    }
    if recipe.mode == "matched" and (not all(checks.values()) or not all(stream_checks.values())):
        raise ValueError("matched intervention changed an unaffected exogenous draw")
    before_inputs, after_inputs = before.inputs, after.inputs
    before_targets, after_targets = before.targets.double(), after.targets.double()
    if isinstance(causal_runtime(recipe.base), NativeVectorRecipe):
        # Group archives restore contiguous tensors; reduce the same layout before/after load.
        before_inputs, after_inputs = before_inputs.contiguous(), after_inputs.contiguous()
        before_targets, after_targets = before_targets.contiguous(), after_targets.contiguous()
    return {
        "version": "causal-intervention-pair-v1",
        "mode": recipe.mode,
        "changes": json.loads(encode([asdict(c) for c in recipe.interventions])),
        "base_mechanism_identity": record_fingerprint(asdict(recipe.base)),
        "row_identity": {
            "seed": seed,
            "task_id": task_id,
            "indices": list(range(len(before.inputs))),
        },
        "split_equal": torch.equal(before.context_indices, after.context_indices)
        and torch.equal(before.query_indices, after.query_indices),
        "unaffected_exogenous_equal": checks,
        "stream_initial_equal": stream_checks,
        "bypassed_nodes": execution.bypassed,
        "coupling": "same keyed exogenous variates"
        if recipe.mode == "matched"
        else "fresh keyed population; equal row positions do not mean same units",
        "input_mean_change": (after_inputs.mean(0) - before_inputs.mean(0)).cpu().numpy().tolist(),
        "target_mean_change": float(after_targets.mean() - before_targets.mean()),
        "interpretation": (
            "generated interventional comparison, not observational conditioning "
            "or a causal estimate from observations"
        ),
    }


def asdict_json(value: CausalBase) -> Any:
    return json.loads(encode(asdict(value)))
