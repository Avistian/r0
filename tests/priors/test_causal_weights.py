"""Structured projection records through shared decoding, generation and traces."""

import json
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


def test_factorized_weights_retain_the_worked_projection_and_its_construction() -> None:
    controls = LinearRecipe(
        rows=16, context_rows=8, features=2, signal_features=2, active_features=2
    )
    data = asdict(
        CausalRecipe(
            task=controls,
            nodes=(
                CausalNode(bias=(0.0, 0.0), noise_scale=(1.0, 1.0)),
                CausalNode(
                    parents=(0,),
                    weights=((17.0, 20.0), (44.0, 50.0)),
                    bias=(0.0, 0.0),
                    noise_scale=(0.0, 0.0),
                ),
            ),
            observed=((0, 0), (0, 1)),
            target=((1, 0),),
            version="causal-v2",
        )
    )
    # [[1,2],[3,4]] diag(2,.5) [[5,6],[7,8]] = [[17,20],[44,50]].
    data["nodes"][1]["weight_construction"] = {
        "version": "factorized-weights-v1",
        "left": [[1, 2], [3, 4]],
        "diagonal": [2, 0.5],
        "right": [[5, 6], [7, 8]],
    }
    recipe = recipe_from_dict(data)
    task = generate_task(recipe, seed=28, task_id=11)
    expected = 17 * task.inputs[:, 0] + 44 * task.inputs[:, 1]
    torch.testing.assert_close(task.scores[:, 0], expected, rtol=1e-14, atol=1e-14)
    trace = {n["id"]: n for n in explain_generation(task, [0, 15])["nodes"]}
    assert trace["causal-1"]["parameters"]["weight_construction"]["diagonal"] == [2, 0.5]
    assert json.loads(task.mechanism_json)["resource_admission"]["parameters"] == 29
    assert (
        generate_task(
            recipe_from_dict(json.loads(json.dumps(asdict(recipe)))), seed=28, task_id=11
        ).episode()
        == task.episode()
    )
