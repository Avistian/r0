"""Original-row intervention locations, bypasses, innovations and propagated states."""

import json
from collections.abc import Callable, Sequence
from dataclasses import asdict
from typing import Any

import torch

from r0.modules.priors.lib.causal_intervention_decoding import intervention_from_dict
from r0.modules.priors.lib.causal_intervention_execution import InterventionExecution
from r0.modules.priors.lib.causal_intervention_generation import generate_intervened
from r0.modules.priors.lib.causal_intervention_records import causal_runtime
from r0.modules.priors.lib.causal_limits import CausalLimits
from r0.modules.priors.lib.native_causal import NativeCausalNode
from r0.modules.priors.lib.training_task import TrainingTask


def intervention_steps(
    task: TrainingTask,
    indices: Sequence[int],
    add: Callable[..., None],
    *,
    limits: CausalLimits | None,
) -> dict[str, Any]:
    record = json.loads(task.recipe_json)
    recipe = intervention_from_dict(record["recipe"], limits=limits or CausalLimits())
    execution = InterventionExecution(recipe)
    replay = generate_intervened(
        recipe, seed=record["seed"], task_id=record["task_id"], execution=execution
    )
    if not torch.equal(replay.inputs, task.inputs.cpu()) or not torch.equal(
        replay.targets, task.targets.cpu()
    ):
        raise ValueError("saved intervention task differs from original-row replay")
    runtime = causal_runtime(recipe.base)
    selected = list(indices)
    for i, node in enumerate(runtime.nodes):
        equation = node.equation if isinstance(node, NativeCausalNode) else node
        changes = [asdict(c) for c in recipe.interventions if c.node == i]
        for role, value in execution.traces[i].items():
            matrix = value[:, None] if value.ndim == 1 else value
            if i in execution.bypassed or role in (
                "root",
                "exogenous",
                "uniform",
                "forced_probability",
            ):
                parents = []
            elif role == "inputs":
                parents = [f"intervention-{p}-state" for p in equation.parents]
            elif role == "function":
                parents = [f"intervention-{i}-inputs"]
            elif role == "scale":
                parents = [f"intervention-{i}-function"]
            else:
                parents = [f"intervention-{i}-function", f"intervention-{i}-exogenous"]
            add(
                f"intervention-{i}-{role}",
                f"Node {i}: {role}",
                parents,
                "surgical replacement; original equation bypassed"
                if i in execution.bypassed
                else "frozen equation evaluated with declared intervention locations",
                matrix[selected],
                {"changes": changes, "role": role, "original_equation": asdict(equation)},
            )
    add(
        "inputs",
        "Observed intervention readouts",
        [],
        "selected declared readouts; state and captured measurements remain distinct",
        task.inputs.cpu()[selected],
        {},
    )
    add(
        "targets",
        "Intervention target",
        [],
        "declared target with original frozen adapter",
        task.targets.cpu()[selected, None],
        {},
    )
    return {
        "version": "generation-graph-v1",
        "episode_id": task.episode_id,
        "rows": selected,
        "component_count": 0,
        "intervention": execution.metadata(),
    }
