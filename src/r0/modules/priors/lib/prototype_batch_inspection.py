"""Original rows retain count/Gamma, weight and convex-mixture dependency order."""

import json
from collections.abc import Callable, Sequence
from dataclasses import asdict
from typing import Any, cast

import torch

from r0.modules.priors.lib.causal_limits import CausalLimits
from r0.modules.priors.lib.exact_frozen_record import exact_frozen_equal
from r0.modules.priors.lib.functional_causal_evaluation import functional_rng
from r0.modules.priors.lib.network_inspection import network_steps
from r0.modules.priors.lib.prototype_batch_decoding import prototype_batch_network_from_dict
from r0.modules.priors.lib.prototype_batch_network import RealizedPrototypeBatchNetworkRecipe
from r0.modules.priors.lib.prototype_batch_output import generate_prototype_batch_network
from r0.modules.priors.lib.training_task import TrainingTask


def prototype_batch_steps(
    task: TrainingTask,
    indices: Sequence[int],
    add: Callable[..., None],
    *,
    limits: CausalLimits | None,
) -> dict[str, Any]:
    record = json.loads(task.recipe_json)
    recipe = prototype_batch_network_from_dict(record["recipe"], limits=limits or CausalLimits())
    if not isinstance(recipe, RealizedPrototypeBatchNetworkRecipe):
        raise ValueError("saved prototype batch must contain a realized population")
    replay = generate_prototype_batch_network(
        recipe, seed=record["seed"], task_id=record["task_id"]
    )
    for field in (
        "source_task_id",
        "episode_id",
        "task_type",
        "classes",
        "feature_schema",
        "target_origin",
    ):
        if not exact_frozen_equal(getattr(task, field), getattr(replay, field)):
            raise ValueError("saved prototype semantic metadata differs from original-row replay")
    expected_record = json.loads(replay.recipe_json)
    # Implementation identity and schema are semantic. Only explicitly descriptive
    # historical environment values may differ from this CPU replay environment.
    execution = record.get("execution")
    if type(execution) is not dict:
        raise ValueError("saved prototype execution schema differs from original-row replay")
    execution = cast(dict[str, Any], execution)
    if (
        set(execution) != set(expected_record["execution"])
        or type(execution.get("implementation")) is not str
        or execution.get("implementation") != expected_record["execution"]["implementation"]
        or execution.get("device") not in ("cpu", "cuda")
        or any(type(execution.get(key)) is not str for key in ("python", "numpy", "torch"))
        or type(execution.get("cpu_threads")) is not int
        or execution["cpu_threads"] < 1
        or any(
            execution.get(key) is not None and type(execution.get(key)) is not str
            for key in ("cuda", "gpu")
        )
    ):
        raise ValueError("saved prototype execution semantics differ from original-row replay")
    if (
        set(record) != set(expected_record)
        or any(
            json.dumps(record[key], sort_keys=True, allow_nan=False)
            != json.dumps(expected_record[key], sort_keys=True, allow_nan=False)
            for key in expected_record
            if key != "execution"
        )
        or json.dumps(json.loads(task.mechanism_json), sort_keys=True, allow_nan=False)
        != json.dumps(json.loads(replay.mechanism_json), sort_keys=True, allow_nan=False)
    ):
        raise ValueError("saved prototype provenance differs from original-row replay")
    for field in ("inputs", "targets", "scores", "context_indices", "query_indices"):
        actual, expected = getattr(task, field), getattr(replay, field)
        if (
            actual.dtype != expected.dtype
            or actual.layout != expected.layout
            or not torch.equal(expected, actual.cpu())
        ):
            raise ValueError("saved prototype population differs from original-row replay")
    if task.classes != replay.classes or (task.probabilities is None) != (
        replay.probabilities is None
    ):
        raise ValueError("saved prototype target adapter differs from original-row replay")
    if (
        task.probabilities is not None
        and replay.probabilities is not None
        and (
            task.probabilities.dtype != replay.probabilities.dtype
            or task.probabilities.layout != replay.probabilities.layout
            or not torch.equal(task.probabilities.cpu(), replay.probabilities)
        )
    ):
        raise ValueError("saved prototype probabilities differ from original-row replay")
    batch = recipe.population
    raw = batch.raw[-1].array()
    add(
        "prototype-raw",
        "Row weight draws",
        [],
        "retained integer counts or raw Gamma variates",
        torch.tensor(raw[list(indices)]),
        {"law": batch.root.weights.mode, "rows": list(indices)},
    )
    add(
        "prototype-weights",
        "Normalized row weights",
        ["prototype-raw"],
        "counts / trials or Gamma / row sum",
        torch.tensor(batch.draw.weights, dtype=torch.float64)[list(indices)],
        {"count": batch.root.count},
    )
    add(
        "inputs",
        "Batch convex-mixture inputs",
        ["prototype-weights"],
        "row weights @ shared prototype matrix",
        replay.inputs[list(indices)],
        {
            "prototypes": batch.root.prototypes,
            "reference_rows": batch.plan.reference_rows,
            "rho": batch.plan.rho,
            "count_rule": batch.plan.count_rule,
        },
    )
    parents = network_steps(
        recipe.request.source.function,
        replay.inputs,
        prefix="network",
        indices=indices,
        add=add,
        input_parents=["inputs"],
    )
    add(
        "signal",
        "Batch network scores",
        parents,
        "frozen architecture on realized population",
        replay.scores[list(indices)],
        {},
    )
    # Share the existing adapter arithmetic, but retain the raw Gaussian ancestor
    # too. The labels/noise generators are independently keyed by original task.
    from r0.modules.priors.lib.generation_graph import target_steps

    def rng(stream: str):
        return functional_rng(record["seed"], record["task_id"], stream)

    if recipe.task.task_type == "regression":
        add(
            "noise-draw",
            "Original standard normal draw",
            [],
            "independently keyed standard Gaussian observation draw",
            torch.tensor(
                rng("noise").normal(size=recipe.task.rows)[list(indices)], dtype=torch.float64
            ),
            {"stream": "noise", "row_indices": list(indices)},
        )

    def adapter_add(
        key: str,
        label: str,
        parents: list[str],
        formula: str,
        values: torch.Tensor,
        parameters: dict[str, Any],
    ) -> None:
        if key == "noise":
            parents, formula = ["noise-draw"], "noise_ratio * original standard normal draw"
        add(key, label, parents, formula, values, parameters)

    targets = target_steps(
        asdict(recipe.task), replay.scores[list(indices)], indices, rng, adapter_add
    )
    original_targets = task.targets.cpu()[list(indices)]
    if not torch.equal(targets, original_targets):
        raise ValueError("prototype target adapter arithmetic differs from original-row replay")
    add("targets", "Task targets", ["target"], "shared target adapter output", targets[:, None], {})
    return {
        "version": "generation-graph-v1",
        "episode_id": task.episode_id,
        "rows": list(indices),
        "component": None,
        "component_count": 0,
        "signal_max_error": float(
            (replay.scores[list(indices)] - task.scores.cpu()[list(indices)]).abs().max()
        ),
        "target_max_error": float((targets - original_targets).abs().max()),
        "population": {"seed": recipe.seed, "task_id": recipe.task_id, "rows": recipe.task.rows},
    }
