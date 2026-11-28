"""Exact category-set splits with preserved raw categorical observations."""

import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pytest
import torch

from r0.modules import generation as g
from r0.modules import priors as p


@pytest.mark.parametrize("codes", [(), (0,), (0, 2, 5), (-3, 2**53)])
def test_exact_membership_including_unseen_and_noninteger(codes: tuple[int, ...]) -> None:
    activation = p.category_subset_activation(codes)
    values = np.array([[-3.0, 0.0], [1.0, 2.0], [2.5, 5.0], [99.0, float(2**53)]])
    expected = np.isin(values, codes).astype(float)
    np.testing.assert_array_equal(activation.apply(torch.tensor(values)), expected)


def test_invalid_and_unrepresentable_codes() -> None:
    for codes in ((1, 1), (2**53 + 1,), (True,)):
        with pytest.raises(ValueError, match="category|activation"):
            p.category_subset_activation(codes)
    with pytest.raises(ValueError, match="representable"):
        p.category_subset_activation((2**24 + 1,)).apply(torch.tensor([[0.0]], dtype=torch.float32))
    with pytest.raises(ValueError, match="count"):
        p.CausalActivation("category-subset-v1", (("count", 2), ("category_0", 1)))


def recipe() -> p.FunctionalCausalRecipe:
    forest = p.RecursiveForest(
        1,
        (
            p.RecursiveTree(
                (
                    p.RecursiveTreeNode(0, 0.5, (0.0, 1.0), 0, 1, 2),
                    p.RecursiveTreeNode(0, 0.0, (0.0, 0.5), 1, value=(-2.0,)),
                    p.RecursiveTreeNode(0, 1.0, (0.5, 1.0), 1, value=(3.0,)),
                )
            ),
        ),
    )
    root = p.SourceRootBase(
        p.SourceRootLaw(
            (
                p.SourceRootCoordinate("categorical", probabilities=(0.1, 0.2, 0.3, 0.4)),
                p.SourceRootCoordinate("gaussian"),
            )
        )
    )
    return p.FunctionalCausalRecipe(
        p.LinearRecipe(rows=64, context_rows=48, features=2, signal_features=2, active_features=2),
        (
            p.FunctionalCausalNode(
                (),
                p.FrozenLinearFunction(((1.0, 0.0), (0.0, 1.0))),
                root,
                version="functional-causal-node-v2",
            ),
            p.FunctionalCausalNode(
                (0,),
                p.FrozenLinearFunction(((1.0, 0.0),)),
                pipeline=p.CausalNodePipeline(1, (p.category_subset_activation((0, 2)),)),
            ),
            p.FunctionalCausalNode((1,), p.FrozenRecursiveForestFunction(forest)),
        ),
        ((0, 0), (0, 1)),
        ((2, 0),),
    )


def test_subset_tree_generation_and_archive(tmp_path: Path) -> None:
    authored = recipe()
    restored = p.recipe_from_dict(json.loads(json.dumps(asdict(authored))))
    assert restored == authored
    task = p.generate_task(restored, seed=29, task_id=2**54 + 1)
    codes = task.inputs[:, 0].numpy()
    assert set(codes) == {0.0, 1.0, 2.0, 3.0}
    np.testing.assert_array_equal(task.targets.numpy(), np.where(np.isin(codes, (0, 2)), 3.0, -2.0))
    path = tmp_path / "subset-tree.zip"
    g.save_group(path, [(2**54 + 1, task)], max_bytes=4 * 1024**2)
    saved = next(g.load_group(path))
    assert saved.recipe_json == task.recipe_json
    assert p.explain_generation(saved, [63, 0, 63])["signal_max_error"] == 0
