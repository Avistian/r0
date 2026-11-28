"""Independent soft path arithmetic, model leaves and standalone persistence."""

from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import pytest
import torch

from r0.modules import generation as g
from r0.modules import priors as p


def test_oblique_split_linear_and_smooth_leaf_oracle() -> None:
    left = p.NetworkAffine(((2.0, 0.0),), (-1.0,))
    right = p.NetworkAffine(((0.0, 3.0),), (0.2,), activation=p.CausalActivation("tanh"))
    f = p.compile_soft_tree(2, (p.SoftTreeBranch((1.0, -2.0), 0.5, 1, 2, 0.3), left, right))
    x = np.array([[0.5, 0.0], [2.0, -1.0], [-1.0, 3.0]])
    gate = 1 / (1 + np.exp(-(x[:, 0] - 2 * x[:, 1] - 0.5) / 0.3))
    expected = (1 - gate) * (2 * x[:, 0] - 1) + gate * np.tanh(3 * x[:, 1] + 0.2)
    actual = p.evaluate_frozen_function(f, torch.tensor(x))
    np.testing.assert_allclose(actual[:, 0], expected, atol=1e-14)
    assert p.frozen_function_from_dict(asdict(f)) == f


def test_deep_tree_path_products_and_vector_leaves() -> None:
    def leaf(a: float, b: float) -> p.NetworkAffine:
        return p.NetworkAffine(((0.0,), (0.0,)), (a, b))

    f = p.compile_soft_tree(
        1,
        (
            p.SoftTreeBranch((1.0,), 0.0, 1, 2),
            leaf(-2.0, 1.0),
            p.SoftTreeBranch((1.0,), 1.0, 3, 4, 0.5),
            leaf(3.0, 4.0),
            leaf(7.0, -1.0),
        ),
    )
    x = np.array([[-2.0], [0.0], [2.0]])
    a = 1 / (1 + np.exp(-x))
    b = 1 / (1 + np.exp(-(x - 1) / 0.5))
    expected = (1 - a) * np.array([-2.0, 1.0]) + a * (
        (1 - b) * np.array([3.0, 4.0]) + b * np.array([7.0, -1.0])
    )
    np.testing.assert_allclose(p.evaluate_frozen_function(f, torch.tensor(x)), expected, atol=1e-14)


def test_temperature_hard_limit_and_equality() -> None:
    leaves = (p.NetworkAffine(((0.0,),), (-2.0,)), p.NetworkAffine(((0.0,),), (3.0,)))
    f = p.compile_soft_tree(1, (p.SoftTreeBranch((1.0,), 0.25, 1, 2, 1e-5), *leaves))
    x = torch.tensor([[0.24], [0.25], [0.26]], dtype=torch.float64)
    np.testing.assert_allclose(p.evaluate_frozen_function(f, x)[:, 0], [-2.0, 0.5, 3.0], atol=1e-14)


def test_invalid_tree_and_limits() -> None:
    leaf = p.NetworkAffine(((0.0,),), (1.0,))
    with pytest.raises(ValueError, match="reachable"):
        p.compile_soft_tree(1, (leaf, leaf))
    with pytest.raises(ValueError, match="follow"):
        p.compile_soft_tree(1, (p.SoftTreeBranch((1.0,), 0.0, 0, 1), leaf))
    with pytest.raises(ValueError, match="temperature"):
        p.SoftTreeBranch((1.0,), 0.0, 1, 2, 0.0)
    with pytest.raises(ValueError, match="limit"):
        p.compile_soft_tree(1, (leaf,), limits=replace(p.CausalLimits(), parameters=1))


def test_soft_tree_standalone_archive(tmp_path: Path) -> None:
    f = p.compile_soft_tree(
        1,
        (
            p.SoftTreeBranch((1.0,), 0.25, 1, 2),
            p.NetworkAffine(((0.0,),), (-2.0,)),
            p.NetworkAffine(((1.0,),), (3.0,)),
        ),
    )
    recipe = p.NetworkRecipe(
        p.LinearRecipe(rows=16, context_rows=8, features=1, signal_features=1, active_features=1), f
    )
    task = p.generate_task(recipe, seed=29, task_id=2**54 + 1)
    path = tmp_path / "soft-tree.zip"
    g.save_group(path, [(2**54 + 1, task)], max_bytes=4 * 1024**2)
    saved = next(g.load_group(path))
    assert saved.recipe_json == task.recipe_json
    expected = p.evaluate_frozen_function(f, task.inputs)[:, 0]
    torch.testing.assert_close(task.targets, expected)
    assert p.explain_generation(saved, [15, 0, 15])["signal_max_error"] < 1e-12
