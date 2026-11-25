"""Sampled settings are ordinary replayable recipes at the public priors seam."""

from dataclasses import asdict

import pytest
import torch

from r0.modules.episodes import TaskType
from r0.modules.priors import (
    LinearRecipe,
    PolynomialRecipe,
    generate_task,
    recipe_from_dict,
    sample_recipe,
)


def test_broad_polynomial_settings_replay_independently_of_call_order():
    task = LinearRecipe(rows=32, context_rows=20, features=16, signal_features=16)
    recipes: list[PolynomialRecipe] = []
    for i in range(32):
        selected = sample_recipe("polynomial", task=task, seed=27, task_id=i)
        assert isinstance(selected, PolynomialRecipe)
        recipes.append(selected)
    assert len({recipe.degree for recipe in recipes}) >= 10
    assert all(recipe.task == task for recipe in recipes)
    assert {recipe.inputs.distribution for recipe in recipes} == {
        "gaussian",
        "uniform",
        "student_t",
        "mixture",
    }
    for i in reversed(range(32)):
        assert sample_recipe("polynomial", task=task, seed=27, task_id=i) == recipes[i]
    recipe = recipes[0]
    original = generate_task(recipe, seed=27, task_id=0)
    replay = generate_task(recipe_from_dict(asdict(recipe)), seed=27, task_id=0)
    assert original.recipe_json == replay.recipe_json
    assert torch.equal(original.inputs, replay.inputs)
    assert torch.equal(original.targets, replay.targets)


def test_broad_portfolio_generates_valid_tasks_without_family_specific_consumer_glue():
    families = (
        "linear",
        "grouped",
        "polynomial",
        "quadratic",
        "rff",
        "periodic",
        "radial",
        "sinusoidal",
        "gp",
        "composition",
    )
    kinds: tuple[tuple[TaskType, int], ...] = (("regression", 0), ("binary", 2), ("multiclass", 3))
    for kind, classes in kinds:
        task = LinearRecipe(rows=32, context_rows=20, task_type=kind, classes=classes)
        for i, family in enumerate(families):
            recipe = sample_recipe(family, task=task, seed=31, task_id=i)
            assert recipe_from_dict(asdict(recipe)) == recipe
            result = generate_task(recipe, seed=31, task_id=i)
            episode = result.episode()
            assert episode.task_type == kind
            assert len(episode.context) == 20
            assert len(episode.queries) == 12
            assert len(episode.features) == 8
            assert bool(torch.isfinite(result.scores).all())
            assert bool(torch.isfinite(result.targets).all())


def test_reference_settings_are_explicit_adaptations_not_broad_limits():
    from r0.modules.priors import Combination, CompositeRecipe, FourierRecipe, QuadraticRecipe

    task = LinearRecipe(
        rows=32, context_rows=20, features=32, signal_features=32, active_features=20
    )
    for task_id in range(8):
        power = sample_recipe(
            "power", task=task, seed=9, task_id=task_id, profile="tabiclv2-adapted-v1"
        )
        assert isinstance(power, CompositeRecipe)
        assert isinstance(power.expression, Combination)
        assert power.expression.operation == "product"
        assert power.expression.children in ((0, 0), (0, 0, 0), (0, 0, 0, 0), (0, 0, 0, 0, 0))
        quadratic = sample_recipe(
            "quadratic", task=task, seed=9, task_id=task_id, profile="tabiclv2-adapted-v1"
        )
        assert isinstance(quadratic, QuadraticRecipe)
        assert quadratic.task.active_features <= 20
        fourier = sample_recipe(
            "rff", task=task, seed=9, task_id=task_id, profile="tabiclv2-adapted-v1"
        )
        assert isinstance(fourier, FourierRecipe)
        assert fourier.terms == 256
        product = sample_recipe(
            "product", task=task, seed=9, task_id=task_id, profile="tabiclv2-adapted-v1"
        )
        assert isinstance(product, CompositeRecipe)
        assert isinstance(product.expression, Combination)
        assert product.expression.operation == "product"
        assert product.expression.children == (0, 1)
        assert len(product.components) == 2
    for family in ("power", "quadratic", "rff", "product"):
        recipe = sample_recipe(family, task=task, seed=9, task_id=0, profile="tabiclv2-adapted-v1")
        result = generate_task(recipe, seed=9, task_id=0)
        assert bool(torch.isfinite(result.scores).all())


def test_profile_exclusions_fail_without_resizing_or_substitution():
    wide = LinearRecipe(features=256, signal_features=256, active_features=256)
    with pytest.raises(ValueError, match="at most 64 active"):
        sample_recipe("periodic", task=wide, seed=0, task_id=0)
    with pytest.raises(ValueError, match="at most 20 active"):
        sample_recipe("quadratic", task=wide, seed=0, task_id=0, profile="tabiclv2-adapted-v1")
    with pytest.raises(ValueError, match="at most 1024 rows"):
        sample_recipe("gp", task=LinearRecipe(rows=20000), seed=0, task_id=0)
    with pytest.raises(ValueError, match="unsupported"):
        sample_recipe("gp", task=wide, seed=0, task_id=0, profile="tabiclv2-adapted-v1")
    with pytest.raises(ValueError, match="seed and task_id"):
        sample_recipe("linear", task=wide, seed=True, task_id=0)
    selected = sample_recipe("quadratic", task=wide, seed=0, task_id=0)
    assert not isinstance(selected, LinearRecipe)
    assert selected.task.active_features == 256
