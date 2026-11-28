"""Converter post functions obey enclosing graph limits before evaluation."""

from dataclasses import asdict, replace

import numpy as np
import pytest
import torch

from r0.modules import priors as p


def recipe_for(
    post: p.FrozenRFFFunction | p.FrozenDiscretizationFunction, limits: p.CausalLimits
) -> p.ConvertedCausalRecipe:
    converter = p.CategoricalConverter(
        "category", 0, 1, ("a", "b"), "neighbor_func", ((-1.0,), (1.0,)), post_function=post
    )
    root = p.ConvertedCausalNode(
        p.FunctionalCausalNode(
            (), p.FrozenLinearFunction(((1.0,),)), p.JointRootBase("gaussian", 1)
        ),
        (converter,),
    )
    child = p.ConvertedCausalNode(p.FunctionalCausalNode((0,), p.FrozenLinearFunction(((1.0,),))))
    return p.ConvertedCausalRecipe(
        p.LinearRecipe(rows=16, context_rows=8, features=1, signal_features=1, active_features=1),
        (root, child),
        (p.ConvertedReadout(0, capture="category"),),
        p.ConvertedReadout(1, channel=0),
        limits,
    )


def test_authored_converter_rejects_nested_channel_width_before_generation() -> None:
    post = p.FrozenRFFFunction(((1.0,) * 8,), (0.1,) * 8, ((0.25,),) * 8)
    limits = p.CausalLimits(node_channels=2)
    with pytest.raises(ValueError, match="channel"):
        p.evaluate_frozen_function(post, torch.ones((16, 1), dtype=torch.float64), limits=limits)
    with pytest.raises(ValueError, match="channel"):
        recipe_for(post, limits)


def test_valid_post_function_roundtrip_and_aggregate_workspace() -> None:
    post = p.FrozenRFFFunction(((1.0,) * 8,), (0.1,) * 8, ((0.25,),) * 8)
    recipe = recipe_for(post, p.CausalLimits(node_channels=8))
    task = p.generate_task(recipe, seed=3, task_id=5)
    index = task.inputs[:, 0].numpy().astype(int)
    selected = np.asarray([-1.0, 1.0])[index]
    expected = (np.cos(selected[:, None] * np.ones(8) + 0.1) @ np.full((8, 1), 0.25))[:, 0]
    np.testing.assert_allclose(task.targets.numpy(), expected, rtol=1e-14)
    assert p.recipe_from_dict(asdict(recipe), limits=recipe.limits) == recipe
    # This is above the old 27 KiB report but below the nested function allocation.
    with pytest.raises(ValueError, match="workspace|memory|storage"):
        replace(recipe, limits=replace(recipe.limits, working_bytes=30_000))


def test_post_function_center_row_floor_is_admitted_for_short_query() -> None:
    post = p.FrozenDiscretizationFunction(
        tuple((float(i),) for i in range(64)), p.FrozenLinearFunction(((1.0,),))
    )
    limits = p.CausalLimits(node_channels=128, working_bytes=300_000)
    # Sixteen query rows still need the function's 64-row center workspace.
    with pytest.raises(ValueError, match="working memory"):
        p.evaluate_frozen_function(post, torch.ones((16, 1), dtype=torch.float64), limits=limits)
    with pytest.raises(ValueError, match="working memory"):
        recipe_for(post, limits)


def test_multiple_post_functions_share_one_aggregate_budget() -> None:
    post = p.FrozenRFFFunction(((1.0,) * 8,), (0.1,) * 8, ((0.25,),) * 8)
    limits = p.CausalLimits(node_channels=8, working_bytes=60_000)
    assert p.generate_task(recipe_for(post, limits), seed=1, task_id=2).inputs.shape == (16, 1)
    converter = p.CategoricalConverter(
        "first", 0, 1, ("a", "b"), "neighbor_func", ((-1.0,), (1.0,)), post_function=post
    )
    root = p.ConvertedCausalNode(
        p.FunctionalCausalNode(
            (), p.FrozenLinearFunction(((1.0,), (-1.0,))), p.JointRootBase("gaussian", 1)
        ),
        (converter, replace(converter, name="second", start=1, stop=2)),
    )
    child = p.ConvertedCausalNode(
        p.FunctionalCausalNode((0,), p.FrozenLinearFunction(((1.0, 1.0),)))
    )
    with pytest.raises(ValueError, match="workspace"):
        p.ConvertedCausalRecipe(
            p.LinearRecipe(
                rows=16, context_rows=8, features=1, signal_features=1, active_features=1
            ),
            (root, child),
            (p.ConvertedReadout(0, capture="first"),),
            p.ConvertedReadout(1, channel=0),
            limits,
        )
