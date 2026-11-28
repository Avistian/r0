"""Mixed predicate trees preserve literal routing, equality and sampled replay."""

import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pytest
import torch

from r0.modules import generation as g
from r0.modules import priors as p


def root() -> p.SourceRootBase:
    return p.SourceRootBase(
        p.SourceRootLaw(
            (
                p.SourceRootCoordinate("gaussian"),
                p.SourceRootCoordinate("categorical", probabilities=(0.2, 0.3, 0.5)),
            )
        )
    )


def task() -> p.LinearRecipe:
    return p.LinearRecipe(
        rows=32, context_rows=16, features=2, signal_features=2, active_features=2
    )


def authored(equality: str) -> p.FunctionalCausalRecipe:
    tree = p.RecursiveTree(
        (
            p.RecursiveTreeNode(0, 0.0, (-1.0, 1.0), 0, 1, 2),
            p.RecursiveTreeNode(0, 0.0, (-1.0, 0.0), 1, value=(0.0,)),
            p.RecursiveTreeNode(1, 0.5, (0.0, 1.0), 1, 3, 4),
            p.RecursiveTreeNode(0, 0.0, (0.0, 1.0), 2, value=(0.0,)),
            p.RecursiveTreeNode(0, 0.0, (0.0, 1.0), 2, value=(0.0,)),
        )
    )
    return p.mixed_model_tree_recipe(
        task(),
        (p.NumericProjectionSplit((1.0, 0.0)), p.CategorySubsetSplit(1, (0, 2))),
        tree,
        tuple(p.NetworkAffine(((1.0, 0.0),), (bias,)) for bias in (10.0, 20.0, 30.0)),
        root=root(),
        equality=equality,
    )


@pytest.mark.parametrize("equality", ["left", "right"])
def test_literal_mixed_routing_at_boundaries_and_unseen_codes(
    equality: str, tmp_path: Path
) -> None:
    recipe = authored(equality)
    x = torch.tensor(
        [[-1.0, 0.0], [0.0, 1.0], [0.0, 2.0], [1.0, 0.0], [1.0, 1.0], [1.0, 2.0], [1.0, 99.0]],
        dtype=torch.float64,
    )
    states = [x]
    for node in recipe.nodes[1:]:
        value = p.evaluate_frozen_function(
            node.function, torch.cat([states[i] for i in node.parents], dim=1)
        )
        if node.pipeline:
            for stage in node.pipeline.stages:
                value = stage.apply(value)
        states.append(value)
    expected: list[float] = []
    for numeric, code in x.numpy():
        left = numeric <= 0 if equality == "left" else numeric < 0
        expected.append(numeric + (10 if left else 30 if code in (0, 2) else 20))
    np.testing.assert_allclose(states[-1].numpy()[:, 0], expected, atol=1e-14)
    assert p.recipe_from_dict(json.loads(json.dumps(asdict(recipe)))) == recipe
    generated = p.generate_task(recipe, seed=29, task_id=0)
    archive = tmp_path / "mixed.zip"
    g.save_group(archive, [(0, generated)], max_bytes=8000000)
    saved = next(g.load_group(archive))
    assert saved.episode() == generated.episode()
    assert p.explain_generation(saved, [31, 0, 31])["signal_max_error"] < 1e-12


@pytest.mark.parametrize("family", ["mixed", "mixed-pruned"])
@pytest.mark.parametrize("equality", ["left", "right"])
def test_sampled_mixed_shapes_replay_and_numeric_columns(family: str, equality: str) -> None:
    request = p.TreeExtensionRequest(
        task(), families=(family,), root=root(), min_depth=2, max_depth=3, equality=equality
    )
    found: set[str] = set()
    for seed in range(4):
        sampled = p.sample_tree_extension(request, rng=np.random.default_rng(seed))
        sampled.validate_replay()
        restored = p.sampled_tree_extension_from_dict(json.loads(json.dumps(asdict(sampled))))
        assert restored == sampled
        assert isinstance(sampled.recipe, p.FunctionalCausalRecipe)
        predicates = sampled.recipe.nodes[1].function
        assert isinstance(predicates, p.FrozenNetworkFunction)
        for block in predicates.blocks[:-1]:
            affine = block.experts[0]
            if affine.activation.kind == "identity":
                assert affine.matrix[0][1] == 0
                found.add("numeric")
            else:
                found.add("categorical")
        generated = p.generate_task(sampled.recipe, seed=29, task_id=seed)
        assert p.explain_generation(generated, [0, 31])["signal_max_error"] < 1e-12
    assert found == {"numeric", "categorical"}


def test_mixed_root_validation_and_depth_zero() -> None:
    with pytest.raises(ValueError, match="categorical"):
        p.TreeExtensionRequest(task(), families=("mixed",))
    categories = p.SourceRootBase(
        p.SourceRootLaw((p.SourceRootCoordinate("categorical", probabilities=(1.0,)),) * 2)
    )
    with pytest.raises(ValueError, match="numeric"):
        p.TreeExtensionRequest(task(), families=("mixed",), root=categories)
    result = p.sample_tree_extension(
        p.TreeExtensionRequest(task(), families=("mixed",), root=root(), min_depth=0, max_depth=0),
        rng=np.random.default_rng(1),
    )
    result.validate_replay()
    assert p.generate_task(result.recipe, seed=29, task_id=0).inputs.shape == (32, 2)
