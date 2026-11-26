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


def test_kernel_points_retain_their_covariance_construction_and_reject_geometry_edits() -> None:
    import pytest

    decay = 0.006737946999085467
    data = {
        "version": "causal-v2",
        "task": {
            "rows": 16,
            "context_rows": 8,
            "features": 2,
            "signal_features": 2,
            "active_features": 2,
        },
        "observed": ((0, 0), (0, 1)),
        "target": ((1, 0),),
        "nodes": [
            {"bias": (0.0, 0.0), "noise_scale": (1.0, 1.0)},
            {
                "parents": (0,),
                "weights": ((1.0, -decay), (-decay, 1.0)),
                "bias": (0.0, 0.0),
                "noise_scale": (0.0, 0.0),
                "weight_construction": {
                    "version": "laplace-kernel-weights-v1",
                    "left_points": ((0.0, 0.0), (3.0, 4.0)),
                    "right_points": ((0.0, 0.0), (3.0, 4.0)),
                    "signs": ((1.0, -1.0), (-1.0, 1.0)),
                    "point_construction": {
                        "version": "kernel-point-cloud-v1",
                        "law": "random-covariance",
                        "base_law": "gaussian",
                        "centers": ((0.0, 0.0),),
                        "factors": (((1.0, 0.0), (0.0, 1.0)),),
                        "factor_weights": (
                            {"version": "uniform-weight-vector-v1", "draws": (0.5, 0.5)},
                        ),
                        "component_weights": {
                            "version": "uniform-weight-vector-v1",
                            "draws": (0.5,),
                        },
                        "assignments": (0, 0, 0, 0),
                        "innovations": ((0.0, 0.0), (6.0, 8.0), (0.0, 0.0), (6.0, 8.0)),
                        "scale_mean": 0.0,
                        "scale_coefficient": 0.0,
                        "scale_draws": (0.0,),
                    },
                },
            },
        ],
    }
    recipe = recipe_from_dict(data)
    task = generate_task(recipe, seed=28, task_id=16)
    assert json.loads(task.mechanism_json)["resource_admission"]["parameters"] == 56
    assert (
        generate_task(
            recipe_from_dict(json.loads(json.dumps(asdict(recipe)))), seed=28, task_id=16
        ).episode()
        == task.episode()
    )
    modified = json.loads(json.dumps(asdict(recipe)))
    modified["nodes"][1]["weight_construction"]["point_construction"]["innovations"][1][0] = 7.0
    with pytest.raises(ValueError, match="point.*construction|construction.*point"):
        recipe_from_dict(modified)
