"""Response curves fix every other input and evaluate the saved analytic function."""

import json
from typing import Any

import numpy as np
import pytest
import torch

from r0.modules.explorer import inspect_curve
from r0.modules.priors import GaussianProcessRecipe, LinearRecipe, generate_task


def test_response_curve_matches_recorded_linear_function_without_mutation() -> None:
    task = generate_task(
        LinearRecipe(rows=32, context_rows=20, noise_ratio=0.7), seed=27, task_id=5
    )
    before = task.inputs.clone(), task.targets.clone(), task.scores.clone()
    card = json.loads(task.mechanism_json)
    feature = card["active"][0]
    curve = inspect_curve(task, feature=feature, anchor_row=7, points=17, bounds=(-3.0, 4.0))
    inputs = np.tile(task.inputs[7].numpy(), (17, 1))
    inputs[:, feature] = np.linspace(-3, 4, 17)
    expected = inputs[:, : len(card["weights"])] @ np.array(card["weights"])
    np.testing.assert_allclose(curve["scores"], expected, rtol=1e-14, atol=1e-14)
    assert curve["anchor_inputs"] == np.asarray(task.inputs[7].numpy(), dtype=np.float64).tolist()
    assert curve["values"][0] == -3
    assert curve["values"][-1] == 4
    assert curve["kind"] == "analytic_response"
    assert curve["feature_role"] == "active"
    for original, actual in zip(before, (task.inputs, task.targets, task.scores), strict=True):
        torch.testing.assert_close(original, actual, rtol=0, atol=0)
    json.dumps(curve, allow_nan=False)


def test_irrelevant_feature_curve_is_constant_and_default_range_uses_all_rows() -> None:
    task = generate_task(
        LinearRecipe(rows=32, context_rows=20, task_type="multiclass", classes=3),
        seed=27,
        task_id=0,
    )
    active = json.loads(task.mechanism_json)["active"]
    feature = next(i for i in range(task.inputs.shape[1]) if i not in active)
    curve = inspect_curve(task, feature=feature, anchor_row=0, points=9)
    np.testing.assert_allclose(curve["scores"], np.tile(task.scores[0].numpy(), (9, 1)))
    assert curve["feature_role"] == "irrelevant"
    assert curve["bounds"] == [
        float(task.inputs[:, feature].min()),
        float(task.inputs[:, feature].max()),
    ]
    assert curve["range_source"] == "observed_min_max"


def test_curve_limits_and_direct_gp_are_explicit() -> None:
    task = generate_task(LinearRecipe(), seed=27, task_id=0)
    invalid: tuple[dict[str, Any], ...] = (
        {"feature": -1},
        {"feature": 0, "anchor_row": -1},
        {"feature": 0, "points": 513},
        {"feature": 0, "points": True},
        {"feature": 0, "bounds": (2, 1)},
        {"feature": 0, "bounds": (0, float("nan"))},
    )
    for arguments in invalid:
        with pytest.raises(ValueError, match="feature|anchor_row|points|bounds"):
            inspect_curve(task, **arguments)
    gp = generate_task(GaussianProcessRecipe(), seed=27, task_id=0)
    with pytest.raises(ValueError, match="conditional"):
        inspect_curve(gp, feature=0)
