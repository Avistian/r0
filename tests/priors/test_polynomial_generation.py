"""Polynomial truth and input-distribution compatibility at the public seam."""

import json
import math
from dataclasses import replace

import pytest
import torch

from r0.modules.priors import (
    InputDistribution,
    LinearRecipe,
    PolynomialRecipe,
    generate_task,
    recipe_from_dict,
)


@pytest.mark.parametrize("distribution", ["gaussian", "uniform", "student_t", "mixture"])
def test_polynomial_row_reconstructs_for_each_input_distribution(distribution: str) -> None:
    recipe = PolynomialRecipe(inputs=InputDistribution(distribution=distribution, df=9), degree=4)
    task = generate_task(recipe, seed=27, task_id=2)
    card = json.loads(task.mechanism_json)
    raw = math.fsum(
        float(task.inputs[0, j]) ** power * weight[0]
        for j, (power, weight) in enumerate(zip(card["powers"], card["weights"], strict=True))
    )
    expected = (raw - card["center"][0]) / card["scale"][0]
    assert float(task.scores[0, 0]) == pytest.approx(expected, rel=1e-10, abs=1e-10)
    assert task.episode() == generate_task(recipe, seed=27, task_id=2).episode()


def test_heavy_tail_moment_boundary_and_explicit_bounded_option() -> None:
    with pytest.raises(ValueError, match="moment"):
        PolynomialRecipe(inputs=InputDistribution(distribution="student_t", df=8), degree=4)
    recipe = PolynomialRecipe(
        inputs=InputDistribution(distribution="student_t", df=3, bound=2), degree=16
    )
    task = generate_task(recipe, seed=7, task_id=3)
    assert torch.all(task.inputs.abs() <= 2)
    assert not torch.any(task.inputs.abs() == 2)  # Rejection sampling, not clipping.
    assert torch.isfinite(task.scores).all()


def test_distribution_pairs_preserve_function_and_active_inputs() -> None:
    recipe = PolynomialRecipe(inputs=InputDistribution(distribution="mixture"), degree=3)
    first = generate_task(recipe, seed=11, task_id=5)
    wider = generate_task(
        replace(recipe, task=replace(recipe.task, features=12)), seed=11, task_id=5
    )
    noisy = generate_task(
        replace(recipe, task=replace(recipe.task, noise_ratio=2)), seed=11, task_id=5
    )
    assert first.mechanism_json == wider.mechanism_json == noisy.mechanism_json
    torch.testing.assert_close(first.inputs, wider.inputs[:, :8], rtol=0, atol=0)
    torch.testing.assert_close(first.scores, wider.scores, rtol=0, atol=0)
    torch.testing.assert_close(first.scores, noisy.scores, rtol=0, atol=0)


def test_linear_special_case_preserves_classification_controls() -> None:
    task = generate_task(
        PolynomialRecipe(
            degree=1,
            task=LinearRecipe(task_type="binary", classes=2, separation=0, class_offsets=(1000.0,)),
        ),
        seed=3,
        task_id=0,
    )
    assert set(task.episode().context_targets) == {"class-1"}


def test_saved_polynomial_recipe_replays_through_shared_decoder() -> None:
    original = generate_task(PolynomialRecipe(), seed=11, task_id=4)
    recipe = recipe_from_dict(json.loads(original.recipe_json)["recipe"])
    assert original.episode() == generate_task(recipe, seed=11, task_id=4).episode()
