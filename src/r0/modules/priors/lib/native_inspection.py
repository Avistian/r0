"""Original-row inspection of native equations, emissions and direct targets."""

import json
from collections.abc import Callable, Sequence
from dataclasses import asdict
from typing import Any

import torch

from r0.modules.priors.lib.causal_limits import CausalLimits
from r0.modules.priors.lib.frozen_parent_function import FrozenParentFunction
from r0.modules.priors.lib.native_decoding import native_recipe_from_dict
from r0.modules.priors.lib.native_evaluation import evaluate_native_graph
from r0.modules.priors.lib.native_graph_decoding import native_graph_from_dict
from r0.modules.priors.lib.parent_function_inspection import parent_function_steps
from r0.modules.priors.lib.training_task import TrainingTask


def native_steps(
    task: TrainingTask,
    indices: Sequence[int],
    add: Callable[..., None],
    *,
    limits: CausalLimits | None,
) -> dict[str, Any]:
    record, card = json.loads(task.recipe_json), json.loads(task.mechanism_json)
    recipe = (
        native_graph_from_dict(record["recipe"], limits=limits or CausalLimits()).recipe
        if record["recipe"].get("version") == "sampled-native-graph-v1"
        else native_recipe_from_dict(record["recipe"], limits=limits or CausalLimits())
    )
    result = evaluate_native_graph(recipe, seed=record["seed"], task_id=record["task_id"])
    selection = list(indices)
    for i, node in enumerate(recipe.nodes):
        parents = [f"causal-{p}" for p in node.equation.parents]
        equation = "f(concat(one-hot discrete parents, continuous parents)) + structural noise"
        if isinstance(node.equation.function, FrozenParentFunction):
            inputs = torch.cat([result.states[p] for p in node.equation.parents], dim=1)
            parents = parent_function_steps(
                node.equation.function, inputs, node=i, parents=parents, indices=indices, add=add
            )
            equation = (
                f"{node.equation.function.aggregation}(ordered encoded parent members)"
                " + structural noise"
            )
        add(
            f"function-{i}",
            f"Native equation {i}",
            parents,
            equation,
            result.scores[i][selection],
            asdict(node.equation),
        )
        emission_parents = [f"function-{i}"]
        if i in result.uniforms:
            recorded = card["native_emissions"][str(i)]["uniforms"]
            if recorded != result.uniforms[i].numpy().tolist():
                raise ValueError("saved native emission variates disagree with replay")
            add(
                f"uniform-{i}",
                f"Native emission uniform {i}",
                [],
                "U[0,1)",
                result.uniforms[i][selection, None],
                {"stream": ["native-emission", i]},
            )
            emission_parents.append(f"uniform-{i}")
        add(
            f"causal-{i}",
            f"Native variable {i}",
            emission_parents,
            node.domain.emission,
            result.values[i][selection],
            asdict(node.domain),
        )
    inputs = torch.stack(
        [result.values[i][:, j].to(torch.float64) for i, j in recipe.observed], dim=1
    )
    target = result.values[recipe.target[0]][:, recipe.target[1]]
    if not torch.equal(inputs, task.inputs.cpu()) or not torch.equal(target, task.targets.cpu()):
        raise ValueError("saved native task values disagree with replay")
    add(
        "inputs",
        "Typed observed variables",
        [f"causal-{i}" for i, _ in recipe.observed],
        "declared native variable readout",
        inputs[selection],
        {"feature_schema": [asdict(f) for f in task.feature_schema]},
    )
    add(
        "targets",
        "Native target variable",
        [f"causal-{recipe.target[0]}"],
        "direct native variable; no target adapter",
        target[selection, None],
        {
            "target_origin": "native-variable-v1",
            "vocabulary": recipe.nodes[recipe.target[0]].domain.vocabulary,
        },
    )
    return {
        "version": "generation-graph-v1",
        "episode_id": task.episode_id,
        "rows": selection,
        "component": None,
        "component_count": 0,
        "target_origin": "native-variable-v1",
    }
