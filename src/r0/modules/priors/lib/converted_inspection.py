"""Replay original typed captures and retain their causal dependencies in inspection."""

import json
from collections.abc import Callable, Sequence
from dataclasses import asdict
from typing import Any

import torch

from r0.modules.priors.lib.causal_limits import CausalLimits
from r0.modules.priors.lib.converted_decoding import converted_recipe_from_dict
from r0.modules.priors.lib.converted_evaluation import evaluate_converted
from r0.modules.priors.lib.converted_graph_decoding import converted_graph_from_dict
from r0.modules.priors.lib.converted_output import converted_value
from r0.modules.priors.lib.root_populations import RootPopulations
from r0.modules.priors.lib.training_task import TrainingTask


def converted_steps(
    task: TrainingTask,
    indices: Sequence[int],
    add: Callable[..., None],
    *,
    limits: CausalLimits | None,
    root_populations: RootPopulations | None = None,
) -> dict[str, Any]:
    record = json.loads(task.recipe_json)
    data = record["recipe"]
    recipe = (
        converted_graph_from_dict(data, limits=limits or CausalLimits()).recipe
        if data["version"] == "sampled-converted-graph-v1"
        else converted_recipe_from_dict(data, limits=limits or CausalLimits())
    )
    result = evaluate_converted(
        recipe,
        seed=record["seed"],
        task_id=record["task_id"],
        add=add,
        indices=indices,
        root_populations=root_populations,
    )
    inputs = torch.stack(
        [converted_value(result, ref).to(torch.float64) for ref in recipe.observed], dim=1
    )
    target = converted_value(result, recipe.target)
    if not torch.equal(inputs, task.inputs.cpu()) or not torch.equal(target, task.targets.cpu()):
        raise ValueError("saved converted task disagrees with original-row replay")
    expected = {
        f"{i}:{name}": {
            "uniforms": result.uniforms[i, name].numpy().tolist()
            if (i, name) in result.uniforms
            else None,
            "categories": value.categories.numpy().tolist(),
            "probabilities": value.probabilities.numpy().tolist(),
        }
        for (i, name), value in result.emissions.items()
    }
    if json.loads(task.mechanism_json)["converter_emissions"] != expected:
        raise ValueError("saved converter emissions disagree with replay")

    def ref_id(node: int, capture: str | None) -> str:
        return f"pipeline-{node}-category-{capture}-capture" if capture else f"readout-{node}"

    add(
        "observed",
        "Typed observed variables",
        [ref_id(r.node, r.capture) for r in recipe.observed],
        "ordered typed captures and numerical readouts",
        inputs[list(indices)],
        {"references": [asdict(r) for r in recipe.observed]},
    )
    add(
        "target",
        "Direct captured target",
        [ref_id(recipe.target.node, recipe.target.capture)],
        "declared captured variable; no target adapter",
        target[list(indices), None],
        asdict(recipe.target),
    )
    return {
        "family": "converted_causal",
        "version": "generation-graph-v1",
        "recipe_version": recipe.version,
        "episode_id": task.episode_id,
        "rows": list(indices),
        "component": None,
        "component_count": 0,
        "target_origin": "native-variable-v1",
    }
