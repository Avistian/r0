"""Independent quadratic reconstruction and paired controls at the public seam."""

import json
import math
from dataclasses import replace

import pytest
import torch

from r0.modules.episodes import TaskType
from r0.modules.priors import (
    InputDistribution,
    LinearRecipe,
    QuadraticRecipe,
    generate_task,
    recipe_from_dict,
)


@pytest.mark.parametrize(
    ("task_type", "classes"), [("regression", 0), ("binary", 2), ("multiclass", 3)]
)
def test_quadratic_scalar_reconstruction_and_replay(task_type: TaskType, classes: int) -> None:
    recipe = QuadraticRecipe(
        task=LinearRecipe(task_type=task_type, classes=classes), interaction_strength=-0.7
    )
    task = generate_task(recipe, seed=27, task_id=1)
    card = json.loads(task.mechanism_json)
    x = [float(v) for v in task.inputs[0]]
    for c in range(task.scores.shape[1]):
        raw = math.fsum(
            x[j]
            * x[k]
            * math.fsum(
                w[c] * a[j] * a[k] for a, w in zip(card["directions"], card["weights"], strict=True)
            )
            * (1 if j == k else recipe.interaction_strength)
            for j in range(recipe.task.signal_features)
            for k in range(recipe.task.signal_features)
        ) + recipe.linear_strength * math.fsum(
            x[j] * w[c] for j, w in enumerate(card["linear_weights"])
        )
        assert float(task.scores[0, c]) == pytest.approx(
            (raw - card["center"][c]) / card["scale"][c], abs=1e-10
        )
    decoded = recipe_from_dict(json.loads(task.recipe_json)["recipe"])
    assert task.episode() == generate_task(decoded, seed=27, task_id=1).episode()


def test_interaction_pair_preserves_factors_and_diagonal() -> None:
    recipe = QuadraticRecipe()
    full = generate_task(recipe, seed=13, task_id=0)
    diagonal = generate_task(replace(recipe, interaction_strength=0), seed=13, task_id=0)
    a, b = (json.loads(t.mechanism_json) for t in (full, diagonal))
    for key in ("directions", "weights", "linear_weights", "active"):
        assert a[key] == b[key]
    torch.testing.assert_close(full.inputs, diagonal.inputs, rtol=0, atol=0)
    x = [float(v) for v in full.inputs[0]]
    expected_delta = math.fsum(
        2
        * x[j]
        * x[k]
        * math.fsum(
            w[0] * direction[j] * direction[k]
            for direction, w in zip(a["directions"], a["weights"], strict=True)
        )
        for j in range(recipe.task.signal_features)
        for k in range(j + 1, recipe.task.signal_features)
    )
    observed_delta = (
        float(full.scores[0, 0]) * a["scale"][0]
        + a["center"][0]
        - (float(diagonal.scores[0, 0]) * b["scale"][0] + b["center"][0])
    )
    assert observed_delta == pytest.approx(expected_delta, abs=1e-10)
    assert abs(observed_delta) > 1e-6


def test_quadratic_moment_boundary_and_bounded_heavy_tails() -> None:
    with pytest.raises(ValueError, match="moment"):
        QuadraticRecipe(inputs=InputDistribution(distribution="student_t", df=4))
    task = generate_task(
        QuadraticRecipe(inputs=InputDistribution(distribution="student_t", df=2, bound=3)),
        seed=7,
        task_id=4,
    )
    assert torch.isfinite(task.scores).all()
    assert torch.all(task.inputs.abs() < 3)


def test_quadratic_nuisance_and_noise_pairs() -> None:
    recipe = QuadraticRecipe(inputs=InputDistribution(distribution="mixture"))
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
