"""Logical variable, joint score and encoded-state inspection of retained vector tasks."""

import json
from collections.abc import Callable, Sequence
from dataclasses import asdict
from typing import Any

import torch

from r0.modules.priors.lib.causal_limits import CausalLimits
from r0.modules.priors.lib.frozen_parent_function import FrozenParentFunction
from r0.modules.priors.lib.joint_root_base import draw_root_base
from r0.modules.priors.lib.native_vector_decoding import (
    native_vector_graph_from_dict,
    native_vector_recipe_from_dict,
)
from r0.modules.priors.lib.native_vector_evaluation import (
    evaluate_native_vector_graph,
    native_vector_rng,
)
from r0.modules.priors.lib.network_inspection import network_steps
from r0.modules.priors.lib.parent_function_inspection import parent_function_steps
from r0.modules.priors.lib.structural_network import FrozenNetworkFunction
from r0.modules.priors.lib.training_task import TrainingTask


def native_vector_steps(
    task: TrainingTask,
    indices: Sequence[int],
    add: Callable[..., None],
    *,
    limits: CausalLimits | None,
) -> dict[str, Any]:
    record = json.loads(task.recipe_json)
    data = record["recipe"]
    recipe = (
        native_vector_graph_from_dict(data, limits=limits or CausalLimits()).recipe
        if data["version"] == "sampled-native-vector-graph-v1"
        else native_vector_recipe_from_dict(data, limits=limits or CausalLimits())
    )
    values = evaluate_native_vector_graph(recipe, seed=record["seed"], task_id=record["task_id"])
    selected = list(indices)
    inputs = torch.stack([values.values[i][j].to(torch.float64) for i, j in recipe.observed], dim=1)
    target = values.values[recipe.target[0]][recipe.target[1]]
    if not torch.equal(inputs, task.inputs.cpu()) or not torch.equal(target, task.targets.cpu()):
        raise ValueError("saved native vector task values disagree with replay")
    card = json.loads(task.mechanism_json)
    for (i, j), uniform in values.uniforms.items():
        if card["native_emissions"][f"{i}:{j}"]["uniforms"] != uniform.numpy().tolist():
            raise ValueError("saved native vector emission variates disagree with replay")
    for index, node in enumerate(recipe.nodes):
        parents = [f"state-{p}" for p in node.equation.parents]
        if isinstance(node.equation.function, FrozenParentFunction):
            parents = parent_function_steps(
                node.equation.function,
                torch.cat([values.states[p] for p in node.equation.parents], dim=1),
                node=index,
                parents=parents,
                indices=indices,
                add=add,
            )
        if isinstance(node.equation.function, FrozenNetworkFunction):
            if node.equation.root is not None:
                network_inputs = torch.tensor(
                    draw_root_base(
                        node.equation.root,
                        recipe.task.rows,
                        native_vector_rng(record["seed"], record["task_id"], "root", index),
                    ).values,
                    dtype=torch.float64,
                )
                parents = [f"root-{index}"]
                add(
                    parents[0],
                    f"Original vector root input {index}",
                    [],
                    "frozen joint root base",
                    network_inputs[selected],
                    asdict(node.equation.root),
                )
            else:
                network_inputs = torch.cat([values.states[p] for p in node.equation.parents], dim=1)
            parents = network_steps(
                node.equation.function,
                network_inputs,
                prefix=f"network-{index}",
                indices=indices,
                add=add,
                input_parents=parents,
            )
        add(
            f"function-{index}",
            f"Joint vector score equation {index}",
            parents,
            "f(concat(encoded parent vectors)) + joint structural innovation",
            values.scores[index][selected],
            asdict(node.equation),
        )
        members: list[str] = []
        for variable, (domain, span) in enumerate(zip(node.domains, node.spans, strict=True)):
            if (index, variable) in values.uniforms:
                add(
                    f"uniform-{index}-{variable}",
                    f"Emission uniform {index}:{variable}",
                    [],
                    "independent U[0,1) for this logical variable",
                    values.uniforms[index, variable][selected, None],
                    {},
                )
            key = f"variable-{index}-{variable}"
            members.append(key)
            add(
                key,
                f"Native variable {index}:{variable}",
                [f"function-{index}"]
                + ([f"uniform-{index}-{variable}"] if (index, variable) in values.uniforms else []),
                f"{domain.emission}(score slice [{span.score_start}:{span.score_stop}])",
                values.values[index][variable][selected, None],
                {"domain": asdict(domain), "span": asdict(span)},
            )
            if (index, variable) in values.probabilities:
                add(
                    f"probability-{index}-{variable}",
                    f"Emission probabilities {index}:{variable}",
                    [f"function-{index}"],
                    "stochastic softmax or deterministic one-hot probabilities",
                    values.probabilities[index, variable][selected],
                    asdict(domain),
                )
        add(
            f"state-{index}",
            f"Encoded parent state {index}",
            members,
            "concat(continuous values, one-hot categorical blocks)",
            values.states[index][selected],
            {"spans": [asdict(s) for s in node.spans]},
        )
    add(
        "inputs",
        "Observed logical variables",
        [f"variable-{i}-{j}" for i, j in recipe.observed],
        "ordered typed readouts",
        torch.stack(
            [values.values[i][j][selected].to(torch.float64) for i, j in recipe.observed], dim=1
        ),
        {"references": recipe.observed},
    )
    i, j = recipe.target
    add(
        "targets",
        "Direct native target",
        [f"variable-{i}-{j}"],
        "selected logical variable, no adapter",
        values.values[i][j][selected, None],
        {"reference": recipe.target, "domain": asdict(recipe.nodes[i].domains[j])},
    )
    return {
        "version": "generation-graph-v1",
        "episode_id": task.episode_id,
        "rows": selected,
        "component": None,
        "component_count": 0,
        "target_origin": "native-variable-v1",
    }
