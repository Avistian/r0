"""Fitted split tables compose on ordered parent slices without refitting."""

import json
from dataclasses import asdict

import numpy as np
import pytest
import torch

from r0.modules import priors as p


@pytest.mark.parametrize("aggregation", ["sum", "product", "mean", "max", "logsumexp"])
@pytest.mark.parametrize("joint", [False, True])
def test_fitted_parent_and_joint_members(aggregation: str, joint: bool) -> None:
    table = p.FittedSplitTable(
        (1, -2, -2),
        (0.5, 0.0, 0.0),
        (1, -1, -1),
        (2, -1, -1),
        ((0.0,), (-2.0,), (3.0,)),
    )
    fitted = p.FrozenFittedTreeFunction(2, (p.FittedTreeOutput((table,), "mean"),))
    joint_table = p.FittedSplitTable(
        (2, -2, -2),
        (0.0, 0.0, 0.0),
        (1, -1, -1),
        (2, -1, -1),
        ((0.0,), (5.0,), (-7.0,)),
    )
    joint_function = (
        p.FrozenFittedTreeFunction(3, (p.FittedTreeOutput((joint_table,), "mean"),))
        if joint
        else None
    )
    function = p.FrozenParentFunction(
        (fitted, p.FrozenLinearFunction(((2.0,),))), aggregation, joint_function
    )
    x = np.array([[9.0, 0.5, -1.0], [4.0, np.nextafter(0.5, np.inf), 2.0], [3.0, 0.6, 0.0]])
    members = [np.where(x[:, 1].astype(np.float32) <= 0.5, -2.0, 3.0), 2 * x[:, 2]]
    if joint:
        members.insert(0, np.where(x[:, 2].astype(np.float32) <= 0.0, 5.0, -7.0))
    stack = np.stack(members, axis=1)
    expected = {
        "sum": stack.sum(1),
        "product": stack.prod(1),
        "mean": stack.mean(1),
        "max": stack.max(1),
        "logsumexp": np.logaddexp.reduce(stack, axis=1),
    }[aggregation]
    restored = p.frozen_function_from_dict(json.loads(json.dumps(asdict(function))))
    np.testing.assert_allclose(
        p.evaluate_frozen_function(restored, torch.tensor(x))[:, 0], expected
    )
    roots = (
        p.FunctionalCausalNode(
            (), p.FrozenLinearFunction(((1.0, 0.0), (0.0, 1.0))), p.JointRootBase("gaussian", 2)
        ),
        p.FunctionalCausalNode(
            (), p.FrozenLinearFunction(((1.0,),)), p.JointRootBase("gaussian", 1)
        ),
    )
    recipe = p.FunctionalCausalRecipe(
        p.LinearRecipe(rows=16, context_rows=8, features=1, signal_features=1, active_features=1),
        (*roots, p.FunctionalCausalNode((0, 1), function)),
        ((0, 0),),
        ((2, 0),),
    )
    assert p.recipe_from_dict(json.loads(json.dumps(asdict(recipe)))) == recipe
    task = p.generate_task(recipe, seed=51, task_id=7)
    assert p.explain_generation(task, [3, 1, 3])["nodes"]
