"""Scalar reconstruction of additive and projected smooth functions."""

import json
import math
from dataclasses import replace

import pytest
import torch

from r0.modules.episodes import TaskType
from r0.modules.priors import (
    InputDistribution,
    LinearRecipe,
    SmoothRecipe,
    generate_task,
    recipe_from_dict,
)


@pytest.mark.parametrize("family", ["sinusoidal", "radial"])
@pytest.mark.parametrize("projection", ["additive", "ridge"])
@pytest.mark.parametrize(
    ("task_type", "classes"), [("regression", 0), ("binary", 2), ("multiclass", 3)]
)
def test_smooth_scalar_truth_and_replay(
    family: str, projection: str, task_type: TaskType, classes: int
) -> None:
    recipe = SmoothRecipe(
        task=LinearRecipe(task_type=task_type, classes=classes),
        family=family,
        projection=projection,
        terms=8,
        width_spread=3,
    )
    task = generate_task(recipe, seed=27, task_id=2)
    card = json.loads(task.mechanism_json)
    x = [float(v) for v in task.inputs[0]]
    values: list[float] = []
    for direction, center, width in zip(
        card["directions"], card["centers"], card["widths"], strict=True
    ):
        z = (
            math.fsum(
                v * a for v, a in zip(x[: recipe.task.signal_features], direction, strict=True)
            )
            - center
        ) / width
        values.append(math.sin(z) if family == "sinusoidal" else math.exp(-z * z / 2))
        if projection == "additive":
            assert sum(a != 0 for a in direction) == 1
        else:
            assert sum(a != 0 for a in direction) == recipe.task.active_features
    for c in range(task.scores.shape[1]):
        raw = math.fsum(v * w[c] for v, w in zip(values, card["weights"], strict=True))
        assert float(task.scores[0, c]) == pytest.approx(
            (raw - card["center"][c]) / card["scale"][c], abs=1e-10
        )
    replay = recipe_from_dict(json.loads(task.recipe_json)["recipe"])
    assert task.episode() == generate_task(replay, seed=27, task_id=2).episode()


@pytest.mark.parametrize("family", ["sinusoidal", "radial"])
@pytest.mark.parametrize("projection", ["additive", "ridge"])
@pytest.mark.parametrize("distribution", ["gaussian", "uniform", "student_t", "mixture"])
def test_bounded_functions_accept_all_input_distributions(
    family: str, projection: str, distribution: str
) -> None:
    task = generate_task(
        SmoothRecipe(
            family=family,
            projection=projection,
            inputs=InputDistribution(distribution=distribution, df=1),
        ),
        seed=11,
        task_id=3,
    )
    assert torch.isfinite(task.scores).all()
    assert torch.isfinite(task.targets).all()


@pytest.mark.parametrize("family", ["sinusoidal", "radial"])
@pytest.mark.parametrize("projection", ["additive", "ridge"])
def test_smooth_controlled_pairs_preserve_function_inputs_and_split(
    family: str, projection: str
) -> None:
    recipe = SmoothRecipe(family=family, projection=projection)
    original = generate_task(recipe, seed=27, task_id=3)
    wide = generate_task(
        replace(recipe, task=replace(recipe.task, features=12)), seed=27, task_id=3
    )
    noisy = generate_task(
        replace(recipe, task=replace(recipe.task, noise_ratio=2)), seed=27, task_id=3
    )
    assert original.mechanism_json == wide.mechanism_json == noisy.mechanism_json
    torch.testing.assert_close(original.inputs, wide.inputs[:, :8], rtol=0, atol=0)
    for paired in (wide, noisy):
        torch.testing.assert_close(original.scores, paired.scores, rtol=0, atol=0)
        torch.testing.assert_close(original.context_indices, paired.context_indices, rtol=0, atol=0)


def test_insufficient_additive_terms_and_nonfinite_arguments_fail_visibly() -> None:
    with pytest.raises(ValueError, match="one term per active"):
        SmoothRecipe(terms=3)
    with pytest.raises(ValueError, match="nonfinite"):
        generate_task(SmoothRecipe(length_scale=5e-324), seed=27, task_id=0)
