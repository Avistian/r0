"""Root and structural-noise laws through public generation and original-row traces."""

import json
import math
from dataclasses import asdict

import torch

from r0.modules.priors import (
    CausalNode,
    CausalRecipe,
    LinearRecipe,
    explain_generation,
    generate_task,
    recipe_from_dict,
)


def test_uniform_root_law_is_saved_replayed_and_named_in_the_trace() -> None:
    controls = LinearRecipe(
        rows=64, context_rows=8, features=1, signal_features=1, active_features=1
    )
    data = asdict(
        CausalRecipe(
            task=controls,
            nodes=(CausalNode(), CausalNode()),
            observed=((0, 0),),
            target=((1, 0),),
            version="causal-v2",
        )
    )
    data["nodes"][0]["exogenous"] = {
        "inputs": {"distribution": "uniform", "location": 0.5, "scale": 1 / math.sqrt(12)}
    }
    recipe = recipe_from_dict(json.loads(json.dumps(data)))
    task = generate_task(recipe, seed=28, task_id=6)
    assert torch.all((task.inputs >= 0) & (task.inputs <= 1))
    assert float(task.inputs.min()) < 0.1
    assert float(task.inputs.max()) > 0.9
    restored = recipe_from_dict(json.loads(json.dumps(asdict(recipe))))
    assert generate_task(restored, seed=28, task_id=6).episode() == task.episode()
    trace = explain_generation(task, [0, 63])
    nodes = {n["id"]: n for n in trace["nodes"]}
    assert nodes["exogenous-0"]["parameters"]["distribution"]["inputs"]["distribution"] == "uniform"
    assert nodes["exogenous-0"]["values"] == nodes["innovation-0"]["values"]
    assert nodes["causal-0"]["values"][0][0] == float(task.inputs[0, 0])
    card = json.loads(task.mechanism_json)
    assert "recorded exogenous" in card["innovation"]
