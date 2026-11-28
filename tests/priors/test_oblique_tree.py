"""Hard oblique boundary routing with independent explicit split arithmetic."""

import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pytest
import torch

from r0.modules import generation as g
from r0.modules import priors as p


def stump() -> p.RecursiveForest:
    return p.RecursiveForest(
        1,
        (
            p.RecursiveTree(
                (
                    p.RecursiveTreeNode(0, 0.25, (-4.0, 4.0), 0, 1, 2),
                    p.RecursiveTreeNode(0, 0.0, (-4.0, 0.25), 1, value=(-2.0,)),
                    p.RecursiveTreeNode(0, 1.0, (0.25, 4.0), 1, value=(3.0,)),
                )
            ),
        ),
    )


def task() -> p.LinearRecipe:
    return p.LinearRecipe(rows=16, context_rows=8, features=2, signal_features=2, active_features=2)


@pytest.mark.parametrize("equality", ["left", "right"])
def test_exact_boundary_and_archive(equality: str, tmp_path: Path) -> None:
    recipe = p.oblique_tree_recipe(task(), ((1.0, -2.0),), stump(), equality=equality)
    x = torch.tensor([[0.24, 0.0], [0.25, 0.0], [0.26, 0.0], [2.25, 1.0]], dtype=torch.float64)
    projected = p.evaluate_frozen_function(recipe.nodes[1].function, x)
    actual = p.evaluate_frozen_function(recipe.nodes[2].function, projected)
    expected = [-2.0, -2.0 if equality == "left" else 3.0, 3.0, -2.0 if equality == "left" else 3.0]
    np.testing.assert_array_equal(actual[:, 0], expected)
    restored = p.recipe_from_dict(json.loads(json.dumps(asdict(recipe))))
    assert restored == recipe
    generated = p.generate_task(restored, seed=29, task_id=2**54 + 1)
    z = generated.inputs.numpy()[:, 0] - 2 * generated.inputs.numpy()[:, 1]
    left = z <= 0.25 if equality == "left" else z < 0.25
    np.testing.assert_array_equal(generated.targets.numpy(), np.where(left, -2.0, 3.0))
    path = tmp_path / "oblique.zip"
    g.save_group(path, [(2**54 + 1, generated)], max_bytes=4 * 1024**2)
    saved = next(g.load_group(path))
    assert saved.recipe_json == generated.recipe_json
    assert p.explain_generation(saved, [15, 0, 15])["signal_max_error"] == 0


def test_repeated_direction_ancestor_intervals_survive_right_equality() -> None:
    forest = p.RecursiveForest(
        1,
        (
            p.RecursiveTree(
                (
                    p.RecursiveTreeNode(0, 0.0, (-4.0, 4.0), 0, 1, 2),
                    p.RecursiveTreeNode(0, -1.0, (-4.0, 0.0), 1, 3, 4),
                    p.RecursiveTreeNode(0, 1.0, (0.0, 4.0), 1, value=(5.0,)),
                    p.RecursiveTreeNode(0, -2.0, (-4.0, -1.0), 2, value=(-2.0,)),
                    p.RecursiveTreeNode(0, -0.5, (-1.0, 0.0), 2, value=(3.0,)),
                )
            ),
        ),
    )
    recipe = p.oblique_tree_recipe(task(), ((1.0, 0.0),), forest, equality="right")
    x = torch.tensor([[-2.0, 99.0], [-1.0, -99.0], [0.0, 5.0], [1.0, 0.0]], dtype=torch.float64)
    z = p.evaluate_frozen_function(recipe.nodes[1].function, x)
    np.testing.assert_array_equal(
        p.evaluate_frozen_function(recipe.nodes[2].function, z)[:, 0], [-2.0, 3.0, 5.0, 5.0]
    )


@pytest.mark.parametrize("equality", ["left", "right"])
def test_adjacent_float_boundary(equality: str) -> None:
    recipe = p.oblique_tree_recipe(task(), ((1.0, 0.0),), stump(), equality=equality)
    x = torch.tensor(
        [[np.nextafter(0.25, -np.inf), 0.0], [0.25, 0.0], [np.nextafter(0.25, np.inf), 0.0]],
        dtype=torch.float64,
    )
    z = p.evaluate_frozen_function(recipe.nodes[1].function, x)
    expected = [-2.0, -2.0 if equality == "left" else 3.0, 3.0]
    np.testing.assert_array_equal(
        p.evaluate_frozen_function(recipe.nodes[2].function, z)[:, 0], expected
    )


def test_oblique_geometry_rejected() -> None:
    with pytest.raises(ValueError, match="geometry"):
        p.oblique_tree_recipe(task(), ((1.0,),), stump())
    with pytest.raises(ValueError, match="equality"):
        p.oblique_tree_recipe(task(), ((1.0, 0.0),), stump(), equality="random")
