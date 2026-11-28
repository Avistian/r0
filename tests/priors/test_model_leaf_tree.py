"""Hard leaf selection followed by explicit independent model arithmetic."""

import json
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import pytest
import torch

from r0.modules import generation as g
from r0.modules import priors as p


def tree() -> p.RecursiveTree:
    return p.RecursiveTree(
        (
            p.RecursiveTreeNode(0, 0.25, (-4.0, 4.0), 0, 1, 2),
            p.RecursiveTreeNode(0, 0.0, (-4.0, 0.25), 1, value=(999.0,)),
            p.RecursiveTreeNode(0, 1.0, (0.25, 4.0), 1, value=(-999.0,)),
        )
    )


def recipe(equality: str = "left") -> p.FunctionalCausalRecipe:
    task = p.LinearRecipe(
        rows=64, context_rows=48, features=2, signal_features=2, active_features=2
    )
    models = (
        p.NetworkAffine(((2.0, 0.0),), (-1.0,)),
        p.NetworkAffine(((0.0, 3.0),), (0.2,), activation=p.CausalActivation("tanh")),
    )
    return p.model_leaf_tree_recipe(task, ((1.0, -2.0),), tree(), models, equality=equality)


@pytest.mark.parametrize("equality", ["left", "right"])
def test_generated_model_leaf_values_and_saved_replay(equality: str, tmp_path: Path) -> None:
    authored = recipe(equality)
    restored = p.recipe_from_dict(json.loads(json.dumps(asdict(authored))))
    assert restored == authored
    task = p.generate_task(restored, seed=29, task_id=2**54 + 1)
    x = task.inputs.numpy()
    z = x[:, 0] - 2 * x[:, 1]
    left = z <= 0.25 if equality == "left" else z < 0.25
    expected = np.where(left, 2 * x[:, 0] - 1, np.tanh(3 * x[:, 1] + 0.2))
    np.testing.assert_allclose(task.targets.numpy(), expected, atol=1e-14)
    path = tmp_path / "model-tree.zip"
    g.save_group(path, [(2**54 + 1, task)], max_bytes=8 * 1024**2)
    saved = next(g.load_group(path))
    assert saved.recipe_json == task.recipe_json
    assert p.explain_generation(saved, [63, 0, 63])["signal_max_error"] < 1e-12


@pytest.mark.parametrize("equality", ["left", "right"])
def test_equality_selects_the_declared_model(equality: str) -> None:
    authored = recipe(equality)
    values = [torch.tensor([[0.25, 0.0]], dtype=torch.float64)]
    for node in authored.nodes[1:]:
        x = torch.cat([values[i] for i in node.parents], dim=1)
        value = p.evaluate_frozen_function(node.function, x)
        if node.pipeline:
            for stage in node.pipeline.stages:
                value = stage.apply(value)
        values.append(value)
    expected = -0.5 if equality == "left" else np.tanh(0.2)
    np.testing.assert_allclose(values[-1].numpy(), [[expected]], atol=1e-14)


def test_model_count_and_resource_rejection() -> None:
    task = p.LinearRecipe(rows=16, context_rows=8, features=1, signal_features=1, active_features=1)
    model = p.NetworkAffine(((1.0,),), (0.0,))
    with pytest.raises(ValueError, match="terminal"):
        p.model_leaf_tree_recipe(task, ((1.0,),), tree(), (model,))
    with pytest.raises(ValueError, match="limits"):
        p.model_leaf_tree_recipe(
            task, ((1.0,),), tree(), (model, model), limits=replace(p.CausalLimits(), nodes=4)
        )


def test_vector_model_leaves_use_terminal_record_order() -> None:
    authored_tree = p.RecursiveTree(
        (
            p.RecursiveTreeNode(0, 0.0, (-4.0, 4.0), 0, 1, 2),
            p.RecursiveTreeNode(0, -1.0, (-4.0, 0.0), 1, value=(9.0,)),
            p.RecursiveTreeNode(0, 1.0, (0.0, 4.0), 1, 3, 4),
            p.RecursiveTreeNode(0, 0.5, (0.0, 1.0), 2, value=(9.0,)),
            p.RecursiveTreeNode(0, 2.0, (1.0, 4.0), 2, value=(9.0,)),
        )
    )
    task = p.LinearRecipe(
        rows=16,
        context_rows=8,
        features=1,
        signal_features=1,
        active_features=1,
        task_type="multiclass",
        classes=3,
    )
    models = tuple(
        p.NetworkAffine(((float(i),), (1.0,), (0.0,)), (float(10 * i), -float(i), 0.0))
        for i in (1, 2, 3)
    )
    authored = p.model_leaf_tree_recipe(task, ((1.0,),), authored_tree, models)
    x = torch.tensor([[-1.0], [0.5], [2.0]], dtype=torch.float64)
    states = [x]
    for node in authored.nodes[1:]:
        value = p.evaluate_frozen_function(
            node.function, torch.cat([states[i] for i in node.parents], dim=1)
        )
        if node.pipeline:
            for stage in node.pipeline.stages:
                value = stage.apply(value)
        states.append(value)
    np.testing.assert_allclose(states[-1], [[9.0, -2.0, 0.0], [21.0, -1.5, 0.0], [36.0, -1.0, 0.0]])
    assert p.recipe_from_dict(asdict(authored)) == authored
