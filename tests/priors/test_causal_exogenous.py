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


def test_correlated_exogenous_channels_keep_their_factor_and_shared_ancestry() -> None:
    controls = LinearRecipe(
        rows=32, context_rows=8, features=1, signal_features=1, active_features=1
    )
    data = asdict(
        CausalRecipe(
            task=controls,
            nodes=(CausalNode(bias=(0.0, 0.0), noise_scale=(1.0, 1.0)),),
            observed=((0, 0),),
            target=((0, 1),),
            version="causal-v2",
        )
    )
    data["nodes"][0]["exogenous"] = {"factor": [[1, 2], [0, 0]]}
    recipe = recipe_from_dict(data)
    task = generate_task(recipe, seed=28, task_id=0)
    torch.testing.assert_close(task.scores[:, 0], 2 * task.inputs[:, 0], rtol=0, atol=0)
    card = json.loads(task.mechanism_json)
    assert "disconnected_target" not in card["flags"]
    assert card["nodes"][0]["exogenous"]["factor"] == [[1, 2], [0, 0]]
    trace = explain_generation(task, [1, 31])
    assert trace["target_max_error"] == 0
    assert (
        generate_task(
            recipe_from_dict(json.loads(json.dumps(asdict(recipe)))), seed=28, task_id=0
        ).episode()
        == task.episode()
    )


def test_uniform_volume_ball_roots_obey_radial_law_and_keep_row_prefixes() -> None:
    controls = LinearRecipe(
        rows=2000, context_rows=32, features=2, signal_features=2, active_features=2
    )
    data = asdict(
        CausalRecipe(
            task=controls,
            nodes=(CausalNode(bias=(0.0,) * 3, noise_scale=(1.0,) * 3),),
            observed=((0, 0), (0, 1)),
            target=((0, 2),),
            version="causal-v2",
        )
    )
    data["nodes"][0]["exogenous"] = {"geometry": "ball"}
    recipe = recipe_from_dict(data)
    task = generate_task(recipe, seed=28, task_id=5)
    values = torch.cat((task.inputs, task.scores), dim=1)
    radii = values.square().sum(dim=1).sqrt()
    assert torch.all(radii <= 1 + 1e-14)
    # For uniform volume in dimension3, R**3 ~ Uniform(0,1), E[R**2]=3/5.
    assert abs(float((radii**3).mean()) - 0.5) < 0.04
    assert abs(float((radii**2).mean()) - 0.6) < 0.04
    assert torch.all(values.mean(dim=0).abs() < 0.04)
    data["task"]["rows"] = 2100
    longer = generate_task(recipe_from_dict(data), seed=28, task_id=5)
    torch.testing.assert_close(task.inputs, longer.inputs[:2000], rtol=0, atol=0)
    assert explain_generation(task, [0, 1999])["target_max_error"] < 1e-14
