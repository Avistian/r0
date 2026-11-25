"""Whole-group sparsity and independently reconstructed linear targets."""

import json
import math
from dataclasses import replace

import pytest
import torch

from r0.modules.episodes import TaskType
from r0.modules.priors import (
    GroupedLinearRecipe,
    InputDistribution,
    LinearRecipe,
    generate_task,
    recipe_from_dict,
)


@pytest.mark.parametrize("distribution", ["gaussian", "uniform", "student_t", "mixture"])
@pytest.mark.parametrize(
    ("task_type", "classes"), [("regression", 0), ("binary", 2), ("multiclass", 3)]
)
def test_whole_groups_and_scalar_truth(
    distribution: str, task_type: TaskType, classes: int
) -> None:
    recipe = GroupedLinearRecipe(
        task=LinearRecipe(task_type=task_type, classes=classes),
        inputs=InputDistribution(distribution=distribution, df=3),
        group_size=2,
    )
    task = generate_task(recipe, seed=27, task_id=3)
    card = json.loads(task.mechanism_json)
    active = set(card["active"])
    assert len(active) == recipe.task.active_features
    assert card["groups"] == [[0, 1], [2, 3], [4, 5], [6, 7]]
    for group in card["groups"]:
        assert len(active.intersection(group)) in (0, len(group))
    for j, weights in enumerate(card["weights"]):
        assert all((weight != 0) == (j in active) for weight in weights)
    for c in range(task.scores.shape[1]):
        raw = math.fsum(float(task.inputs[0, j]) * w[c] for j, w in enumerate(card["weights"]))
        expected = (raw - card["center"][c]) / card["scale"][c]
        assert float(task.scores[0, c]) == pytest.approx(expected, abs=1e-10)
    decoded = recipe_from_dict(json.loads(task.recipe_json)["recipe"])
    assert task.episode() == generate_task(decoded, seed=27, task_id=3).episode()


def test_grouped_noise_and_nuisance_pairs_preserve_mechanism() -> None:
    recipe = GroupedLinearRecipe()
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


def test_partial_groups_and_infinite_variance_are_rejected_explicitly() -> None:
    with pytest.raises(ValueError, match="whole multiples"):
        GroupedLinearRecipe(group_size=3)
    with pytest.raises(ValueError, match="moment"):
        GroupedLinearRecipe(inputs=InputDistribution(distribution="student_t", df=2))
    bounded = generate_task(
        GroupedLinearRecipe(inputs=InputDistribution(distribution="student_t", df=1, bound=3)),
        seed=27,
        task_id=0,
    )
    assert torch.isfinite(bounded.scores).all()
    assert torch.all(bounded.inputs.abs() < 3)


@pytest.mark.parametrize("group_size", [1, 4, 8])
def test_group_sizes_and_small_scale_preserve_valid_tasks(group_size: int) -> None:
    recipe = GroupedLinearRecipe(
        task=LinearRecipe(active_features=8),
        group_size=group_size,
        inputs=InputDistribution(scale=1e-15),
    )
    task = generate_task(recipe, seed=27, task_id=0)
    unit = generate_task(replace(recipe, inputs=InputDistribution()), seed=27, task_id=0)
    torch.testing.assert_close(task.scores, unit.scores, rtol=1e-10, atol=1e-10)
