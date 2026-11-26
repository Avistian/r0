"""Structured projection records through shared decoding, generation and traces."""

import json
import math
from dataclasses import asdict

import pytest
import torch

from r0.modules.priors import (
    CausalNode,
    CausalRecipe,
    ImportanceWeights,
    LinearRecipe,
    MatrixCausalNode,
    explain_generation,
    generate_task,
    recipe_from_dict,
    sample_recipe,
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
                    weights=((17.0, 20.0), (44.0, 52.0)),
                    bias=(0.0, 0.0),
                    noise_scale=(0.0, 0.0),
                ),
            ),
            observed=((0, 0), (0, 1)),
            target=((1, 0),),
            version="causal-v2",
        )
    )
    # [[1,2],[3,4]] diag(2,.5) [[5,6],[7,8]] = [[17,20],[44,52]].
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
    assert trace["causal-1"]["parameters"]["weight_construction"]["diagonal"] == (2, 0.5)
    assert json.loads(task.mechanism_json)["resource_admission"]["parameters"] == 29
    assert (
        generate_task(
            recipe_from_dict(json.loads(json.dumps(asdict(recipe)))), seed=28, task_id=11
        ).episode()
        == task.episode()
    )


def test_sampled_factorized_record_cannot_silently_discard_its_construction() -> None:
    recipe = sample_recipe(
        "redirection",
        task=LinearRecipe(rows=16, context_rows=8),
        seed=28,
        task_id=10,
        profile="causal-factorized-v1",
    )
    data = asdict(recipe)
    child = next(node for node in data["nodes"] if node["parents"])
    child.pop("weight_construction")
    with pytest.raises(ValueError, match="factorized.*construction"):
        recipe_from_dict(data)


@pytest.mark.parametrize(
    ("left", "diagonal", "right", "expected"),
    [
        (((2.0**1023,),), (4.0,), ((2.0**-1023,),), 4.0),
        (((2.0**-1022,),), (2.0**-60,), ((2.0**1023,),), 2.0**-59),
        (((2.0**1023, 2.0**1023),), (4.0, 4.0), ((1.0,), (-1.0,)), 0.0),
    ],
)
def test_representable_factor_products_survive_extreme_intermediates(
    left: tuple[tuple[float, ...], ...],
    diagonal: tuple[float, ...],
    right: tuple[tuple[float, ...], ...],
    expected: float,
) -> None:
    recipe = CausalRecipe(
        task=LinearRecipe(
            rows=16, context_rows=8, features=1, signal_features=1, active_features=1
        ),
        nodes=(CausalNode(), CausalNode(parents=(0,), weights=((expected,),), noise_scale=(0.0,))),
        observed=((0, 0),),
        target=((1, 0),),
        version="causal-v2",
    )
    data = asdict(recipe)
    data["nodes"][1]["weight_construction"] = {"left": left, "diagonal": diagonal, "right": right}
    task = generate_task(recipe_from_dict(data), seed=28, task_id=11)
    torch.testing.assert_close(task.scores[:, 0], expected * task.inputs[:, 0], rtol=0, atol=0)


def test_uniform_importance_weights_retain_their_normalized_projection() -> None:
    recipe = recipe_from_dict(
        {
            "version": "causal-v2",
            "task": {
                "rows": 16,
                "context_rows": 8,
                "features": 2,
                "signal_features": 2,
                "active_features": 2,
            },
            "observed": [[0, 0], [0, 1]],
            "target": [[1, 0]],
            "nodes": [
                {"bias": [0, 0], "noise_scale": [1, 1]},
                {
                    "parents": [0],
                    "weights": [[0.6, 0.8], [0, 0]],
                    "bias": [0, 0],
                    "noise_scale": [0, 0],
                    "weight_construction": {
                        "version": "importance-weights-v1",
                        "gaussian": [[3, 4], [0, 0]],
                        "importance": [{"draws": [0.5, 0.5]}, {"draws": [0.1, 0.2]}],
                        "epsilon": 0,
                    },
                },
            ],
        }
    )
    task = generate_task(recipe, seed=28, task_id=12)
    torch.testing.assert_close(task.scores[:, 0], 0.6 * task.inputs[:, 0], rtol=0, atol=0)
    trace = {n["id"]: n for n in explain_generation(task, [0, 15])["nodes"]}
    assert trace["causal-1"]["parameters"]["weight_construction"]["importance"][0]["draws"] == (
        0.5,
        0.5,
    )
    assert json.loads(task.mechanism_json)["resource_admission"]["parameters"] == 28
    assert (
        generate_task(
            recipe_from_dict(json.loads(json.dumps(asdict(recipe)))), seed=28, task_id=12
        ).episode()
        == task.episode()
    )


def test_factorized_profile_rejects_a_different_matrix_construction() -> None:
    recipe = sample_recipe(
        "redirection",
        task=LinearRecipe(rows=16, context_rows=8),
        seed=28,
        task_id=10,
        profile="causal-factorized-v1",
    )
    data = asdict(recipe)
    child = next(node for node in data["nodes"] if node["parents"])
    width = len(child["bias"])
    zero = [[0] * width for _ in child["weights"]]
    child["weights"] = zero
    child["weight_construction"] = {
        "version": "importance-weights-v1",
        "gaussian": zero,
        "importance": [{"draws": [0.5] * width} for _ in zero],
    }
    with pytest.raises(ValueError, match="factorized.*construction"):
        recipe_from_dict(data)


@pytest.mark.parametrize(
    ("decay", "noise", "draws", "permutation"),
    [
        (1.0, 0.0, [0.0, 0.0], [1, 0]),
        (0.0, 1.0, [0.0, math.log(2)], [0, 1]),
        (1.0, 1e200, [1.0, 1.0], [1, 0]),
    ],
)
def test_polynomial_importance_records_decay_log_noise_and_permutation(
    decay: float, noise: float, draws: list[float], permutation: list[int]
) -> None:
    vector = {
        "version": "polynomial-weight-vector-v1",
        "draws": draws,
        "decay": decay,
        "log_noise_scale": noise,
        "permutation": permutation,
    }
    recipe = recipe_from_dict(
        {
            "version": "causal-v2",
            "task": {
                "rows": 16,
                "context_rows": 8,
                "features": 2,
                "signal_features": 2,
                "active_features": 2,
            },
            "observed": [[0, 0], [0, 1]],
            "target": [[1, 0]],
            "nodes": [
                {"bias": [0, 0], "noise_scale": [1, 1]},
                {
                    "parents": [0],
                    "weights": [[1, 0], [0, 1]],
                    "bias": [0, 0],
                    "noise_scale": [0, 0],
                    "weight_construction": {
                        "version": "importance-weights-v1",
                        "gaussian": [[1, 0], [0, 1]],
                        "importance": [vector, vector],
                        "epsilon": 0,
                    },
                },
            ],
        }
    )
    assert isinstance(recipe, CausalRecipe)
    node = recipe.nodes[1]
    assert isinstance(node, MatrixCausalNode)
    assert isinstance(node.weight_construction, ImportanceWeights)
    # Both independently worked laws give weights with ratio 1:2 and squared norm 2.
    assert node.weight_construction.importance[0].values() == pytest.approx(
        (math.sqrt(2 / 5), math.sqrt(8 / 5)), rel=1e-14
    )
    task = generate_task(recipe, seed=28, task_id=13)
    torch.testing.assert_close(task.scores[:, 0], task.inputs[:, 0], rtol=0, atol=0)
    assert json.loads(task.mechanism_json)["resource_admission"]["parameters"] == 36
    trace = {n["id"]: n for n in explain_generation(task, [0])["nodes"]}
    assert trace["causal-1"]["parameters"]["weight_construction"]["importance"][0][
        "permutation"
    ] == tuple(permutation)
    assert (
        generate_task(
            recipe_from_dict(json.loads(json.dumps(asdict(recipe)))), seed=28, task_id=13
        ).episode()
        == task.episode()
    )


def test_factor_coefficients_retain_their_positive_vector_law() -> None:
    data = asdict(
        CausalRecipe(
            task=LinearRecipe(
                rows=16, context_rows=8, features=2, signal_features=2, active_features=2
            ),
            nodes=(
                CausalNode(bias=(0.0, 0.0), noise_scale=(1.0, 1.0)),
                CausalNode(
                    parents=(0,),
                    weights=((6.0, 7.0), (14.5, 17.0)),
                    bias=(0.0, 0.0),
                    noise_scale=(0.0, 0.0),
                ),
            ),
            observed=((0, 0), (0, 1)),
            target=((1, 0),),
            version="causal-v2",
        )
    )
    data["nodes"][1]["weight_construction"] = {
        "version": "importance-factorized-weights-v1",
        "left": ((1.0, 2.0), (3.0, 4.0)),
        "right": ((5.0, 6.0), (7.0, 8.0)),
        "diagonal": {"version": "uniform-weight-vector-v1", "draws": (0.5, 0.25)},
    }
    recipe = recipe_from_dict(data)
    task = generate_task(recipe, seed=28, task_id=17)
    torch.testing.assert_close(
        task.scores[:, 0], 6 * task.inputs[:, 0] + 14.5 * task.inputs[:, 1], rtol=1e-14, atol=1e-14
    )
    trace = {n["id"]: n for n in explain_generation(task, [0])["nodes"]}
    assert trace["causal-1"]["parameters"]["weight_construction"]["diagonal"]["draws"] == (
        0.5,
        0.25,
    )
    assert json.loads(task.mechanism_json)["resource_admission"]["parameters"] == 29
    assert (
        generate_task(
            recipe_from_dict(json.loads(json.dumps(asdict(recipe)))), seed=28, task_id=17
        ).episode()
        == task.episode()
    )
