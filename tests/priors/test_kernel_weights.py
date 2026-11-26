"""Kernel projection geometry survives public generation, inspection and replay."""

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


def test_signed_laplace_projection_records_points_signs_and_generated_target() -> None:
    # A 3-4-5 separation gives exp(-5), independently tabulated here.
    decay = 0.006737946999085467
    recipe = CausalRecipe(
        task=LinearRecipe(
            rows=16, context_rows=8, features=2, signal_features=2, active_features=2
        ),
        nodes=(
            CausalNode(bias=(0.0, 0.0), noise_scale=(1.0, 1.0)),
            CausalNode(
                parents=(0,),
                weights=((1.0, -decay), (-decay, 1.0)),
                bias=(0.0, 0.0),
                noise_scale=(0.0, 0.0),
            ),
        ),
        observed=((0, 0), (0, 1)),
        target=((1, 0),),
        version="causal-v2",
    )
    data = asdict(recipe)
    data["nodes"][1]["weight_construction"] = {
        "version": "laplace-kernel-weights-v1",
        "left_points": ((0.0, 0.0), (3.0, 4.0)),
        "right_points": ((0.0, 0.0), (3.0, 4.0)),
        "signs": ((1.0, -1.0), (-1.0, 1.0)),
        "distance_scale": 1.0,
    }
    recorded = recipe_from_dict(data)
    task = generate_task(recorded, seed=28, task_id=14)
    torch.testing.assert_close(
        task.scores[:, 0], task.inputs[:, 0] - decay * task.inputs[:, 1], rtol=1e-14, atol=1e-14
    )
    trace = {n["id"]: n for n in explain_generation(task, [0, 15])["nodes"]}
    assert trace["causal-1"]["parameters"]["weight_construction"]["left_points"] == (
        (0.0, 0.0),
        (3.0, 4.0),
    )
    assert json.loads(task.mechanism_json)["resource_admission"]["parameters"] == 32
    assert (
        generate_task(
            recipe_from_dict(json.loads(json.dumps(asdict(recorded)))), seed=28, task_id=14
        ).episode()
        == task.episode()
    )
