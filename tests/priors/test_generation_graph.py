"""Graph values explain recorded generation, without mutating its training task."""

import json
from dataclasses import replace

import numpy as np
import pytest
import torch

from r0.modules.priors import (
    Combination,
    CompositeRecipe,
    FourierRecipe,
    FunctionComponent,
    GaussianProcessRecipe,
    LinearRecipe,
    PolynomialRecipe,
    SmoothRecipe,
    StationaryKernel,
    explain_generation,
    generate_task,
)


def test_linear_graph_reconstructs_targets_and_preserves_original_task() -> None:
    task = generate_task(LinearRecipe(noise_ratio=0.7), seed=27, task_id=7)
    original = task.episode(), task.recipe_json, task.mechanism_json
    graph = explain_generation(task, [2, 9, 47])
    nodes = {node["id"]: node for node in graph["nodes"]}
    weights = np.array(json.loads(task.mechanism_json)["weights"])
    expected = task.inputs.numpy()[[2, 9, 47], : len(weights)] @ weights
    np.testing.assert_allclose(nodes["signal"]["values"], expected)
    np.testing.assert_allclose(nodes["target"]["values"], task.targets.numpy()[[2, 9, 47], None])
    assert graph["target_max_error"] < 1e-12
    assert original == (task.episode(), task.recipe_json, task.mechanism_json)
    json.dumps(graph, allow_nan=False)


def test_composition_graph_exposes_real_children_and_label_draws() -> None:
    controls = LinearRecipe(task_type="multiclass", classes=3)
    recipe = CompositeRecipe(
        task=replace(controls, separation=0, class_offsets=(0, -2, 1)),
        components=(
            FunctionComponent(controls),
            FunctionComponent(PolynomialRecipe(task=controls)),
        ),
        expression=Combination("product", (0, 1)),
    )
    task = generate_task(recipe, seed=27, task_id=0)
    graph = explain_generation(task, [0, 1, 2])
    nodes = {node["id"]: node for node in graph["nodes"]}
    np.testing.assert_allclose(
        np.array(nodes["component-0"]["values"]) * np.array(nodes["component-1"]["values"]),
        nodes["expression-0"]["values"],
    )
    assert [nodes[f"component-{i}"]["component"] for i in range(2)] == [0, 1]
    child = explain_generation(task, [0, 1, 2], component=1)
    np.testing.assert_allclose(child["nodes"][-1]["values"], nodes["component-1"]["values"])
    assert not any(node["id"] == "target" for node in child["nodes"])
    np.testing.assert_array_equal(
        np.array(nodes["target"]["values"])[:, 0], task.targets[:3].numpy()
    )
    assert "uniform" in nodes["labels"]["formula"]


def test_gp_graph_uses_original_joint_draw_and_real_covariance() -> None:
    recipe = GaussianProcessRecipe(
        task=LinearRecipe(rows=16, context_rows=8),
        calibration_rows=32,
        kernels=(StationaryKernel(), StationaryKernel(kind="matern32", length_scale=2)),
    )
    task = generate_task(recipe, seed=27, task_id=0)
    graph = explain_generation(task, [0, 4, 15])
    nodes = {node["id"]: node for node in graph["nodes"]}
    active = json.loads(task.mechanism_json)["active"]
    x = task.inputs.numpy()[[0, 4, 15]][:, active]
    r = np.sqrt(((x[:, None] - x[None, :]) ** 2).sum(2))
    expected = 0.5 * np.exp(-0.5 * r**2) + 0.5 * (1 + np.sqrt(3) * r / 2) * np.exp(
        -np.sqrt(3) * r / 2
    )
    np.testing.assert_allclose(nodes["covariance"]["values"], expected)
    torch.testing.assert_close(
        torch.tensor(nodes["signal"]["values"], dtype=torch.float64), task.scores[[0, 4, 15]]
    )
    assert nodes["joint-draw"]["parameters"]["value_kind"] == "original_joint_draw"
    with pytest.raises(ValueError, match="1..16"):
        explain_generation(task, list(range(17)))
    with pytest.raises(ValueError, match="row"):
        explain_generation(task, [-1])


@pytest.mark.parametrize(
    "recipe",
    [
        FourierRecipe(),
        FourierRecipe(family="periodic", terms=8),
        SmoothRecipe(),
        SmoothRecipe(family="sinusoidal"),
    ],
)
def test_graph_replays_family_specific_noise_namespaces(
    recipe: FourierRecipe | SmoothRecipe,
) -> None:
    recipe = replace(recipe, task=replace(recipe.task, noise_ratio=0.9))
    task = generate_task(recipe, seed=27, task_id=12)
    graph = explain_generation(task, [0, 7, 63])
    assert graph["target_max_error"] < 1e-10
