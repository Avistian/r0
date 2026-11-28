"""Tree helpers preserve root populations and their existing replay custody."""

import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pytest
import torch

from r0.modules import generation as g
from r0.modules import priors as p
from r0.modules.priors.lib.oblique_tree import TreeRoot


def root_for(kind: str) -> TreeRoot:
    if kind == "prototype":
        return p.PrototypeRootBase(
            p.FrozenPrototypeRoot(
                ((-2.0, -4.0), (1.0, 2.0), (3.0, 6.0)),
                p.PrototypeWeightLaw(trials=1),
            )
        )
    if kind == "sampled":
        profile = p.sample_source_root_profile(
            p.SourceRootProfile(2, selector="forced-r0", families=("gaussian", "categorical")),
            model_rng=np.random.default_rng(29),
            selection_rng=np.random.default_rng(30),
        )
        return p.sample_source_root_base(profile, modes=("identity",) * 2)
    laws = {
        "mixed": (
            p.SourceRootCoordinate("unit_uniform"),
            p.SourceRootCoordinate("categorical", probabilities=(0.1, 0.2, 0.7)),
        ),
        "categorical": (
            p.SourceRootCoordinate("categorical", probabilities=(0.5, 0.5)),
            p.SourceRootCoordinate("categorical", probabilities=(0.25, 0.75)),
        ),
        "heavy-tail": (
            p.SourceRootCoordinate("symmetric_uniform"),
            p.SourceRootCoordinate("zipf", exponent=1.5),
        ),
    }
    return p.SourceRootBase(p.SourceRootLaw(laws[kind]))


def tree() -> p.RecursiveTree:
    return p.RecursiveTree(
        (
            p.RecursiveTreeNode(0, 0.25, (-4.0, 4.0), 0, 1, 2),
            p.RecursiveTreeNode(0, 0.0, (-4.0, 0.25), 1, value=(-2.0,)),
            p.RecursiveTreeNode(0, 1.0, (0.25, 4.0), 1, value=(3.0,)),
        )
    )


@pytest.mark.parametrize("kind", ["mixed", "categorical", "heavy-tail", "prototype", "sampled"])
def test_populations_match_between_tree_helpers_and_survive_archive(
    kind: str, tmp_path: Path
) -> None:
    root = root_for(kind)
    task = p.LinearRecipe(
        rows=64, context_rows=48, features=2, signal_features=2, active_features=2
    )
    hard = p.oblique_tree_recipe(task, ((1.0, -2.0),), p.RecursiveForest(1, (tree(),)), root=root)
    model = p.model_leaf_tree_recipe(
        task,
        ((1.0, -2.0),),
        tree(),
        (
            p.NetworkAffine(((2.0, 0.0),), (-1.0,)),
            p.NetworkAffine(((0.0, 3.0),), (0.2,), activation=p.CausalActivation("tanh")),
        ),
        root=root,
    )
    tasks: list[p.TrainingTask] = []
    for index, recipe in enumerate((hard, model)):
        assert recipe.nodes[0].root is root
        assert recipe.nodes[0].version == "functional-causal-node-v2"
        restored = p.recipe_from_dict(json.loads(json.dumps(asdict(recipe))))
        assert restored == recipe
        generated = p.generate_task(restored, seed=29, task_id=2**54 + 1)
        path = tmp_path / f"population-{index}.zip"
        g.save_group(path, [(2**54 + 1, generated)], max_bytes=8 * 1024**2)
        saved = next(g.load_group(path))
        assert saved.recipe_json == generated.recipe_json
        assert p.explain_generation(saved, [63, 0, 63])["signal_max_error"] < 1e-12
        tasks.append(generated)
    torch.testing.assert_close(tasks[0].inputs, tasks[1].inputs, rtol=0, atol=0)
    x = tasks[0].inputs.numpy()
    left = x[:, 0] - 2 * x[:, 1] <= 0.25
    np.testing.assert_array_equal(tasks[0].targets.numpy(), np.where(left, -2.0, 3.0))
    np.testing.assert_allclose(
        tasks[1].targets.numpy(),
        np.where(left, 2 * x[:, 0] - 1, np.tanh(3 * x[:, 1] + 0.2)),
        atol=1e-14,
    )
    if kind == "categorical":
        assert set(np.unique(x)) <= {0.0, 1.0}
    elif kind == "prototype":
        points: list[list[float]] = x.tolist()
        assert {(row[0], row[1]) for row in points} <= {(-2.0, -4.0), (1.0, 2.0), (3.0, 6.0)}
    elif kind == "heavy-tail":
        assert np.all(np.abs(x[:, 0]) <= 1)
        assert np.all(x[:, 1] >= 1)
