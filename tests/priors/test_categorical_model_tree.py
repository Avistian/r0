"""Categorical split laws and independent generated model-leaf arithmetic."""

import copy
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pytest

from r0.modules import generation as g
from r0.modules import priors as p


@pytest.mark.parametrize("probability", [0.0, 0.3, 1.0])
def test_subset_sampling_draw_oracle(probability: float) -> None:
    vocabularies = ((0, 2, 7), (), (-2, 9))
    actual = p.sample_category_subset_splits(
        vocabularies, count=20, inclusion_probability=probability, rng=np.random.default_rng(29)
    )
    oracle = np.random.default_rng(29)
    for split in actual:
        feature = int(oracle.integers(3))
        keep = oracle.random(len(vocabularies[feature])) < probability
        expected = tuple(
            v for v, selected in zip(vocabularies[feature], keep, strict=True) if selected
        )
        assert split == p.CategorySubsetSplit(feature, expected)


def test_invalid_subset_preflight_does_not_draw() -> None:
    rng = np.random.default_rng(29)
    before = copy.deepcopy(rng.bit_generator.state)
    with pytest.raises(ValueError, match="category"):
        p.sample_category_subset_splits(((0, 0),), count=3, rng=rng)
    assert rng.bit_generator.state == before


def categorical_recipe() -> p.FunctionalCausalRecipe:
    task = p.LinearRecipe(
        rows=64, context_rows=48, features=2, signal_features=2, active_features=2
    )
    tree = p.RecursiveTree(
        (
            p.RecursiveTreeNode(0, 0.5, (0.0, 1.0), 0, 1, 2),
            p.RecursiveTreeNode(0, 0.0, (0.0, 0.5), 1, value=(0.0,)),
            p.RecursiveTreeNode(0, 1.0, (0.5, 1.0), 1, value=(0.0,)),
        )
    )
    root = p.SourceRootBase(
        p.SourceRootLaw(
            (
                p.SourceRootCoordinate("categorical", probabilities=(0.1, 0.2, 0.3, 0.4)),
                p.SourceRootCoordinate("gaussian"),
            )
        )
    )
    return p.categorical_model_tree_recipe(
        task,
        (p.CategorySubsetSplit(0, (0, 2)),),
        tree,
        (p.NetworkAffine(((0.0, 2.0),), (-1.0,)), p.NetworkAffine(((0.0, -3.0),), (0.5,))),
        root=root,
    )


def test_categorical_model_outputs_and_archive(tmp_path: Path) -> None:
    recipe = categorical_recipe()
    assert p.recipe_from_dict(json.loads(json.dumps(asdict(recipe)))) == recipe
    task = p.generate_task(recipe, seed=29, task_id=7)
    x = task.inputs.numpy()
    assert set(x[:, 0]) == {0.0, 1.0, 2.0, 3.0}
    expected = np.where(np.isin(x[:, 0], (0, 2)), -3 * x[:, 1] + 0.5, 2 * x[:, 1] - 1)
    np.testing.assert_allclose(task.targets.numpy(), expected, atol=1e-14)
    path = tmp_path / "categories.zip"
    g.save_group(path, [(7, task)], max_bytes=8 * 1024**2)
    saved = next(g.load_group(path))
    np.testing.assert_array_equal(saved.targets.numpy(), task.targets.numpy())
    assert p.explain_generation(saved, [0, 63])["signal_max_error"] < 1e-12
