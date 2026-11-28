"""One generated table exposes paired regression and fixed-threshold class labels."""

import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pytest
import torch

from r0.modules import priors as p
from r0.modules.generation import load_group, save_group


@pytest.mark.parametrize("boundaries", [(0.0,), (-1.0, 0.0, 1.0)])
def test_dual_targets_share_table_and_persist(boundaries: tuple[float, ...], tmp_path: Path):
    recipe = p.PolynomialRecipe(task=p.LinearRecipe(rows=24, context_rows=12), degree=2)
    table = p.generate_dual_targets(recipe, boundaries=boundaries, seed=17, task_id=4)
    regression, classification = table.regression, table.classification
    assert torch.equal(regression.inputs, classification.inputs)
    assert torch.equal(regression.context_indices, classification.context_indices)
    expected = np.searchsorted(boundaries, regression.targets.numpy(), side="right")
    np.testing.assert_array_equal(classification.targets, expected)
    assert classification.classes == tuple(f"class-{i}" for i in range(len(boundaries) + 1))
    saved = tmp_path / "classification.zip"
    save_group(saved, [(4, classification)], max_bytes=1_000_000)
    restored = next(load_group(saved))
    assert torch.equal(restored.targets, classification.targets)
    decoded = p.recipe_from_dict(json.loads(restored.recipe_json)["recipe"])
    replay = p.generate_task(decoded, seed=17, task_id=4)
    assert torch.equal(replay.inputs, regression.inputs)
    assert torch.equal(replay.targets, classification.targets)
    assert asdict(decoded)["boundaries"] == boundaries


def test_boundary_equality_goes_to_upper_class():
    recipe = p.LinearRecipe(rows=16, context_rows=8)
    base = p.generate_task(recipe, seed=3, task_id=5)
    threshold = float(base.targets[0])
    table = p.generate_dual_targets(recipe, boundaries=(threshold,), seed=3, task_id=5)
    assert table.classification.targets[0] == 1


@pytest.mark.parametrize("boundaries", [(), (1.0, 0.0), (0.0, 0.0), (float("nan"),)])
def test_invalid_cutpoints_rejected(boundaries: tuple[float, ...]):
    with pytest.raises(ValueError, match="boundar"):
        p.generate_dual_targets(p.LinearRecipe(), boundaries=boundaries, seed=0, task_id=0)
