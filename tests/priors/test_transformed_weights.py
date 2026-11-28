"""Activation matrices retain flattened coupling and explicit jitter stages."""

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


def test_activation_projection_ranks_the_whole_child_matrix_and_retains_jitter() -> None:
    # Row-normalized (4,3),(2,1) have flattened order [third,first,second,fourth].
    # Their ascending ranks are [[2,1],[3,0]], then add the recorded .25*jitter.
    recipe = CausalRecipe(
        task=LinearRecipe(
            rows=16, context_rows=8, features=2, signal_features=2, active_features=2
        ),
        nodes=(
            CausalNode(bias=(0.0, 0.0), noise_scale=(1.0, 1.0)),
            CausalNode(
                parents=(0,),
                weights=((2.25, 0.75), (3.5, -0.5)),
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
        "version": "transformed-weights-v1",
        "child": {
            "version": "normalized-weights-v1",
            "source": {"version": "gaussian-weights-v1", "draws": ((4.0, 3.0), (2.0, 1.0))},
            "jitter": ((0.0, 0.0), (0.0, 0.0)),
            "epsilon": 0.0,
        },
        "activation": {"kind": "rank"},
        "jitter": ((1.0, -1.0), (2.0, -2.0)),
        "jitter_scale": 0.25,
    }
    recorded = recipe_from_dict(data)
    task = generate_task(recorded, seed=28, task_id=15)
    torch.testing.assert_close(
        task.scores[:, 0],
        2.25 * task.inputs[:, 0] + 3.5 * task.inputs[:, 1],
        rtol=1e-14,
        atol=1e-14,
    )
    trace = {n["id"]: n for n in explain_generation(task, [0, 15])["nodes"]}
    construction = trace["causal-1"]["parameters"]["weight_construction"]
    assert construction["activation"]["kind"] == "rank"
    assert construction["child"]["source"]["draws"] == ((4.0, 3.0), (2.0, 1.0))
    assert (
        generate_task(
            recipe_from_dict(json.loads(json.dumps(asdict(recorded)))), seed=28, task_id=15
        ).episode()
        == task.episode()
    )
