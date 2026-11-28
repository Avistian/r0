"""Standalone response curves expose frozen raw channels, not adapted class probabilities."""

import json
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch

from r0.modules import priors as p
from r0.modules.explorer import ExplorerSession, inspect_curve


def recipe(kind: Any) -> p.NetworkRecipe:
    width = 3 if kind == "multiclass" else 1
    task = p.LinearRecipe(
        rows=16,
        context_rows=8,
        features=2,
        signal_features=2,
        active_features=2,
        task_type=kind,
        classes=0 if kind == "regression" else 2 if kind == "binary" else 3,
        noise_ratio=2 if kind == "regression" else 0,
        separation=1 if kind == "regression" else 9,
        class_offsets=() if kind == "regression" else (8.0,) * width,
    )
    a = p.NetworkAffine(
        tuple((float(i + 1), -0.5) for i in range(width)),
        (0.25,) * width,
        activation=p.CausalActivation("tanh"),
    )
    b = p.NetworkAffine(
        tuple((-0.3, float(i + 1)) for i in range(width)),
        (-0.2,) * width,
        mask=((True, False),) * width,
    )
    gate = p.NetworkAffine(((0.8, -0.2), (-0.1, 0.6)), (0.1, -0.4))
    skip = p.NetworkProjection(0, p.NetworkAffine(((0.2, 0.3),) * width, (0.05,) * width))
    fn = p.FrozenNetworkFunction(2, (p.NetworkBlock((0,), (a, b), gate, (skip,)),))
    return p.NetworkRecipe(task, fn, root=p.JointRootBase("uniform", 2))


def oracle(x: Any, width: int) -> Any:
    logits = x @ np.array([[0.8, -0.2], [-0.1, 0.6]]).T + [0.1, -0.4]
    weights = np.exp(logits - logits.max(axis=1, keepdims=True))
    weights /= weights.sum(axis=1, keepdims=True)
    a = np.tanh(x @ np.array([(i + 1, -0.5) for i in range(width)]).T + 0.25)
    b = np.repeat((-0.3 * x[:, 0] - 0.2)[:, None], width, axis=1)
    skip = (x @ np.array([0.2, 0.3]) + 0.05)[:, None]
    return weights[:, :1] * a + weights[:, 1:] * b + skip


@pytest.mark.parametrize("kind", ["regression", "binary", "multiclass"])
def test_network_curve_raw_channel_oracle_and_edited_recipe(kind: str):
    r = recipe(kind)
    task = p.generate_task(r, seed=5, task_id=7)
    before = task.recipe_json, task.targets.clone()
    curve = inspect_curve(task, feature=0, anchor_row=4, points=19, bounds=(-2, 3))
    x = np.repeat(task.inputs[4:5].numpy(), 19, axis=0)
    x[:, 0] = np.linspace(-2, 3, 19)
    expected = oracle(x, r.function.output_width)
    np.testing.assert_allclose(curve["scores"], expected, rtol=1e-13, atol=1e-13)
    np.testing.assert_allclose(
        p.evaluate_signal(task, task.inputs),
        oracle(task.inputs.numpy(), r.function.output_width),
        rtol=1e-13,
        atol=1e-13,
    )
    assert task.recipe_json == before[0]
    assert torch.equal(task.targets, before[1])
    block = r.function.blocks[0]
    changed = replace(r, network=replace(r.function, blocks=(replace(block, skips=()),)))
    edited = p.generate_task(changed, seed=5, task_id=7)
    after = inspect_curve(edited, feature=0, anchor_row=4, points=19, bounds=(-2, 3))
    np.testing.assert_allclose(
        after["scores"],
        expected - (x @ np.array([0.2, 0.3]) + 0.05)[:, None],
        rtol=1e-13,
        atol=1e-13,
    )


def test_guarded_network_curve_caps_timeout_and_saved_baseline(tmp_path: Path):
    session = ExplorerSession(tmp_path)
    result = session.request(
        "generate", {"recipe": asdict(recipe("multiclass")), "seed": 5, "task_id": 7}
    )
    before = session.task
    files = sorted(str(f) for f in tmp_path.rglob("*"))
    curve = session.request(
        "curve", {"feature": 0, "anchor_row": 4, "points": 19, "bounds": [-2, 3]}
    )["curve"]
    x = np.repeat(np.array(curve["anchor_inputs"])[None, :], 19, axis=0)
    x[:, 0] = curve["values"]
    np.testing.assert_allclose(curve["scores"], oracle(x, 3), rtol=1e-13, atol=1e-13)
    assert "not class probabilities" in curve["score_semantics"]
    for args, message in (
        ({"max_seconds": 0.001}, "timed out"),
        ({"max_bytes": 1}, "byte cap"),
        ({"causal_limits": {"parameters": 1}}, "limit"),
        ({"points": 513}, "points"),
    ):
        with pytest.raises(ValueError, match=message):
            session.request("curve", {"feature": 0, **args})
        assert session.task is before
    assert session.request("view", {})["recipe_json"] == result["recipe_json"]
    assert sorted(str(f) for f in tmp_path.rglob("*")) == files
    json.dumps(curve, allow_nan=False)
