"""Original rows retain count/Gamma, weight and convex-mixture dependency order."""

import json
from collections.abc import Callable, Sequence
from typing import Any

import torch

from r0.modules.priors.lib.causal_limits import CausalLimits
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
    for field in ("inputs", "targets", "scores", "context_indices", "query_indices"):
        if not torch.equal(getattr(replay, field), getattr(task, field).cpu()):
            raise ValueError("saved prototype population differs from original-row replay")
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
    add(
        "targets",
        "Task targets",
        ["signal"],
        "shared target adapter",
        replay.targets[list(indices), None],
        {},
    )
    return {
        "version": "generation-graph-v1",
        "episode_id": task.episode_id,
        "rows": list(indices),
        "component": None,
        "component_count": 0,
        "signal_max_error": 0.0,
        "target_max_error": 0.0,
        "population": {"seed": recipe.seed, "task_id": recipe.task_id, "rows": recipe.task.rows},
    }
