"""Literal enclosing adapters preserve keyed original-row target draws."""

import hashlib
import json
from typing import Any, Literal

import numpy as np
import pytest
import torch

from r0.modules import priors as p


def source(task_type: Literal["regression", "binary", "multiclass"]) -> p.HyperpriorGraph:
    classes = 3 if task_type == "multiclass" else 2 if task_type == "binary" else 0
    width = classes if task_type == "multiclass" else 1
    task = p.LinearRecipe(
        rows=16,
        context_rows=8,
        features=1,
        signal_features=1,
        active_features=1,
        task_type=task_type,
        classes=classes,
        noise_ratio=0.4 if task_type == "regression" else 0.0,
        separation=1.0 if task_type == "regression" else 1.7,
        class_offsets=() if task_type == "regression" else tuple(0.2 * i for i in range(width)),
    )
    function = p.FrozenLinearFunction(
        tuple(tuple(float(i == j) for j in range(width)) for i in range(width))
    )
    request = p.HyperpriorGraphRequest(
        task,
        p.FixedSizeTopologyHyperprior(3),
        p.FunctionalCausalNode((), function, p.JointRootBase("gaussian", width)),
        p.FunctionalCausalNode((0,), function),
    )
    return p.sample_hyperprior_graph(request, rng=np.random.default_rng(7))


def rng(key: str):
    data = json.dumps(
        ("functional-causal-v1", 19, 23, (key,)), sort_keys=True, separators=(",", ":")
    )
    return np.random.default_rng(int(hashlib.sha256(data.encode()).hexdigest(), 16))


@pytest.mark.parametrize("task_type", ["regression", "binary", "multiclass"])
def test_reference_adapter_matches_literal_formula(
    task_type: Literal["regression", "binary", "multiclass"],
):
    recipe = source(task_type)
    scores = torch.tensor([[-1.5], [0.0], [1.3]], dtype=torch.float64)
    if task_type == "multiclass":
        scores = torch.cat((scores, scores + 0.4, -scores), dim=1)
    output = p.topology_reference_targets(recipe, scores, seed=19, task_id=23, row=5)
    if task_type == "regression":
        noise = float(rng("noise").normal(size=16)[5])
        np.testing.assert_allclose(
            output["fixed_draw_targets"], scores[:, 0].numpy() + 0.4 * noise, rtol=0, atol=1e-12
        )
        assert output["target_standard_deviation"] == 0.4
        assert output["target_mean"] == np.asarray(scores[:, 0].numpy(), dtype=np.float64).tolist()
    else:
        logits = 1.7 * scores.numpy() + np.asarray(recipe.task.class_offsets)
        if task_type == "binary":
            pos = 1 / (1 + np.exp(-logits[:, 0]))
            expected = np.column_stack((1 - pos, pos))
        else:
            exponent = np.exp(logits - logits.max(axis=1, keepdims=True))
            expected = exponent / exponent.sum(axis=1, keepdims=True)
        np.testing.assert_allclose(output["target_probabilities"], expected, rtol=0, atol=1e-12)
        uniform = float(rng("labels").uniform(size=16)[5])
        np.testing.assert_array_equal(
            output["fixed_draw_targets"], (uniform >= expected.cumsum(axis=1)[:, :-1]).sum(axis=1)
        )


@pytest.mark.parametrize("task_type", ["regression", "binary", "multiclass"])
def test_reference_original_scores_reproduce_original_targets(
    task_type: Literal["regression", "binary", "multiclass"],
):
    recipe = source(task_type)
    task = p.generate_task(recipe, seed=19, task_id=23)
    scores = task.scores[5:6].repeat(3, 1)
    result = p.topology_reference_targets(recipe, scores, seed=19, task_id=23, row=5)
    assert result["fixed_draw_targets"] == [task.targets[5].item()] * 3
    if task.probabilities is not None:
        np.testing.assert_allclose(
            result["target_probabilities"],
            task.probabilities[5:6].repeat(3, 1).numpy(),
            rtol=0,
            atol=0,
        )


@pytest.mark.parametrize(
    "bad", ["seed-bool", "task-float", "row", "dtype", "width", "count", "nan"]
)
def test_reference_rejects_known_controls_before_restore(bad: str, monkeypatch: pytest.MonkeyPatch):
    recipe = source("regression")
    scores = torch.ones((3, 1), dtype=torch.float64)
    controls: dict[str, Any] = {"seed": 19, "task_id": 23, "row": 5}
    if bad == "seed-bool":
        controls["seed"] = True
    elif bad == "task-float":
        controls["task_id"] = 23.0
    elif bad == "row":
        controls["row"] = 16
    elif bad == "dtype":
        scores = scores.float()
    elif bad == "width":
        scores = scores.repeat(1, 2)
    elif bad == "count":
        scores = scores[:1]
    else:
        scores[0, 0] = float("nan")

    def forbidden(*args: object, **kwargs: object):
        pytest.fail("known invalid controls restored a construction stream")

    monkeypatch.setattr(type(recipe.topology.initial), "restore", forbidden)
    with pytest.raises(ValueError, match="reference"):
        p.topology_reference_targets(recipe, scores, **controls)
