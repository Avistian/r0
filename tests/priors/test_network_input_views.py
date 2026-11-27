"""Input views preserve joint roots and retain the unchanged network archive."""

import json
from dataclasses import asdict, replace

import numpy as np
import pytest
import torch

from r0.modules import priors as p


def base(root: str = "ball") -> p.NetworkRecipe:
    function = p.FrozenNetworkFunction(
        2, (p.NetworkBlock((0,), (p.NetworkAffine(((2.0, -3.0),), (0.5,)),)),)
    )
    return p.NetworkRecipe(
        p.LinearRecipe(rows=32, context_rows=16, features=2, signal_features=2, active_features=2),
        function,
        p.JointRootBase(root, 2),
    )


@pytest.mark.parametrize("root", ["ball", "gaussian", "uniform"])
def test_extra_columns_preserve_original_joint_draw_and_targets(root: str) -> None:
    original = base(root)
    before = p.generate_task(original, seed=28, task_id=9007199254740993)
    view = p.NetworkInputView(original, replace(original.task, features=5), (0, 1))
    after = p.generate_task(view, seed=28, task_id=9007199254740993)
    assert torch.equal(before.inputs, after.inputs[:, :2])
    assert torch.equal(before.scores, after.scores)
    assert torch.equal(before.targets, after.targets)
    assert torch.equal(before.context_indices, after.context_indices)
    wider = p.generate_task(
        replace(view, task=replace(view.task, features=7)), seed=28, task_id=9007199254740993
    )
    assert torch.equal(after.inputs, wider.inputs[:, :5])
    assert (
        json.loads(after.recipe_json)["recipe"]["base"] == json.loads(before.recipe_json)["recipe"]
    )
    restored = p.recipe_from_dict(json.loads(after.recipe_json)["recipe"])
    assert asdict(restored) == asdict(view)
    replay = p.generate_task(restored, seed=28, task_id=9007199254740993)
    assert replay.recipe_json == after.recipe_json
    assert torch.equal(replay.inputs, after.inputs)


def test_mask_is_after_root_and_signal_accepts_observed_coordinates() -> None:
    original = base()
    view = p.NetworkInputView(original, replace(original.task, features=4, active_features=1), (1,))
    task = p.generate_task(view, seed=9, task_id=7)
    np.testing.assert_array_equal(task.scores.numpy()[:, 0], -3 * task.inputs.numpy()[:, 1] + 0.5)
    changed = task.inputs.clone()
    changed[:, 0] = 123.0
    changed[:, 2:] = -500.0
    assert torch.equal(p.evaluate_signal(task, changed), task.scores)
    assert json.loads(task.mechanism_json)["active"] == [1]


def test_view_rejects_unused_counts_and_bad_masks() -> None:
    original = base()
    for active in ((), (True,), (2,), (0, 0), (1, 0)):
        with pytest.raises(ValueError, match="network.*view"):
            p.NetworkInputView(original, original.task, active)
    with pytest.raises(ValueError, match="network.*view"):
        p.NetworkInputView(original, original.task, (0,))
