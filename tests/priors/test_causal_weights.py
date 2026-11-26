"""Structured projection records through shared decoding, generation and traces."""

import json
from dataclasses import asdict

import pytest
import torch

from r0.modules.priors import (
    CausalNode,
    CausalRecipe,
    LinearRecipe,
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
