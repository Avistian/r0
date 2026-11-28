"""Whole categorical task construction, draw laws and guarded archive receipts."""

import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pytest

from r0.modules import generation as g
from r0.modules import priors as p
from r0.modules.generation.lib.construction import GraphConstruction


def request(family: str, depth: int) -> p.TreeExtensionRequest:
    root = p.SourceRootBase(
        p.SourceRootLaw(
            (
                p.SourceRootCoordinate("gaussian"),
                p.SourceRootCoordinate("categorical", probabilities=(0.1, 0.2, 0.3, 0.4)),
            )
        )
    )
    return p.TreeExtensionRequest(
        p.LinearRecipe(rows=32, context_rows=16, features=2, signal_features=2, active_features=2),
        families=(family,),
        min_depth=depth,
        max_depth=depth,
        root=root,
    )


@pytest.mark.parametrize("family", ["categorical", "categorical-pruned"])
@pytest.mark.parametrize("depth", [0, 1, 3])
def test_categorical_complete_receipt(family: str, depth: int) -> None:
    sampled = p.sample_tree_extension(request(family, depth), rng=np.random.default_rng(29))
    restored = p.sampled_tree_extension_from_dict(json.loads(json.dumps(asdict(sampled))))
    assert restored == sampled
    task = p.generate_task(restored.recipe, seed=29, task_id=7)
    assert {float(v) for v in task.inputs[:, 1]} == {0.0, 1.0, 2.0, 3.0}
    assert p.explain_generation(task, [0, 31])["signal_max_error"] < 1e-12


def test_membership_draw_order_excludes_continuous_columns() -> None:
    sampled = p.sample_tree_extension(request("categorical", 1), rng=np.random.default_rng(29))
    assert isinstance(sampled.recipe, p.FunctionalCausalRecipe)
    function = sampled.recipe.nodes[1].function
    assert isinstance(function, p.FrozenNetworkFunction)
    split = function.blocks[0].experts[0]
    assert split.matrix == ((0.0, 1.0),)
    oracle = np.random.default_rng(29)
    oracle.integers(1)
    oracle.integers(1, 2)
    oracle.integers(1)
    categories = tuple(i for i, included in enumerate(oracle.random(4) < 0.5) if included)
    assert split.activation == p.category_subset_activation(categories)


def test_categorical_request_requires_raw_category_root() -> None:
    with pytest.raises(ValueError, match="categorical columns"):
        p.TreeExtensionRequest(p.LinearRecipe(), families=("categorical",))


def test_categorical_per_task_guarded_archive(tmp_path: Path) -> None:
    controls = request("categorical-pruned", 2)
    run = g.RunRequest(
        controls.task,
        seed=29,
        count=2,
        max_seconds=30,
        max_bytes=8000000,
        construction=GraphConstruction(
            json.dumps({"topology": "tree-extension", "request": asdict(controls)}), 42
        ),
    )
    result = g.run_guarded(run, tmp_path, workers=2)
    assert result.returncode == 0, result.stderr
    assert result.summary is not None
    assert result.summary["completed"] == 2
    for path in tmp_path.rglob("data-*.zip"):
        for task in g.load_group(path):
            raw = json.loads(task.recipe_json)["construction"]["sampled"]
            p.sampled_tree_extension_from_dict(raw)
    resumed = g.run_guarded(run, tmp_path)
    assert resumed.summary is not None
    assert resumed.summary["generated"] == 0
