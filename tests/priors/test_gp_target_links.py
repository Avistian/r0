"""Linked GP target edits preserve exact parents and fixed stochastic environments."""

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch

from r0.modules import generation as g
from r0.modules import priors as p


def environment(kind: Any, mode: str):
    classes = {"regression": 0, "binary": 2, "multiclass": 3}[kind]
    outputs = 3 if classes == 3 else 1
    population = p.realize_gp_columns(
        p.GPColumnRequest(rows=16, outputs=outputs), seed=23, task_id=2**54 + 19
    )
    task = p.LinearRecipe(
        rows=16,
        context_rows=8,
        features=2,
        signal_features=2,
        active_features=2,
        task_type=kind,
        classes=classes,
        noise_ratio=0.4 if not classes else 0.0,
    )
    target = p.GPColumnTarget("linear", ((0.7,) * outputs, (-0.2,) * outputs), (0.1,) * outputs)
    pair = p.generate_gp_column_pair(
        p.GPColumnRecipe(population, task),
        target,
        mode=mode,
        fresh_seed=2**54 + 23 if mode == "fresh" else None,
    )
    recipe = p.recipe_from_dict(json.loads(pair.after.recipe_json)["recipe"])
    assert isinstance(recipe, p.GPColumnTargetRecipe)
    return recipe, pair.after


@pytest.mark.parametrize("kind", ["regression", "binary", "multiclass"])
@pytest.mark.parametrize("mode", ["matched", "fresh"])
def test_two_link_literal_calibration_parent_exactness_and_archive(
    kind: Any, mode: str, tmp_path: Path
):
    base, original = environment(kind, mode)
    outputs = base.request.outputs
    quadratic = p.GPColumnTarget(
        "quadratic",
        ((0.4,) * outputs, (0.2,) * outputs),
        (0.0,) * outputs,
        quadratic=tuple(((0.1, 0.3), (-0.2, 0.4)) for _ in range(outputs)),
    )
    first = p.generate_gp_link_pair(base, quadratic)
    assert first.before.recipe_json == original.recipe_json
    assert torch.equal(first.before.targets, original.targets)
    parent = p.recipe_from_dict(json.loads(first.after.recipe_json)["recipe"])
    assert isinstance(parent, p.GPColumnTargetLink)
    neural = p.GPColumnTarget(
        "neural",
        ((0.2,) * outputs, (0.7,) * outputs),
        (0.3,) * outputs,
        hidden=((0.4, -0.3), (0.2, 0.8)),
        hidden_bias=(0.2, -0.4),
    )
    second = p.generate_gp_link_pair(parent, neural)
    assert second.before.recipe_json == first.after.recipe_json
    assert torch.equal(second.before.targets, first.after.targets)
    assert torch.equal(second.after.inputs, original.inputs)
    assert torch.equal(second.after.context_indices, original.context_indices)
    raw = json.loads(second.after.recipe_json)["recipe"]
    assert raw["parent"] == json.loads(first.after.recipe_json)["recipe"]
    x = second.after.inputs.numpy()
    cal = np.asarray(base.comparison.original.calibration_inputs)
    fit = np.tanh(cal @ neural.hidden + neural.hidden_bias) @ neural.weights + neural.bias
    values = np.tanh(x @ neural.hidden + neural.hidden_bias) @ neural.weights + neural.bias
    expected = (values - fit.mean(axis=0)) / fit.std(axis=0)
    np.testing.assert_allclose(second.after.scores.numpy(), expected, atol=1e-13)
    if kind == "regression":
        np.testing.assert_allclose(
            (second.after.targets - second.after.scores[:, 0]).numpy(),
            (original.targets - original.scores[:, 0]).numpy(),
            atol=1e-15,
        )
    g.save_causal_pair(tmp_path / "pair", second, max_bytes=128 * 1024**2)
    reopened = g.load_causal_pair(tmp_path / "pair")
    assert reopened.after.recipe_json == second.after.recipe_json
    assert p.explain_generation(reopened.after, [5, 2, 5])["rows"] == [5, 2, 5]
    decoded = p.gp_column_target_link_from_dict(raw)
    reference = p.evaluate_gp_environment_reference(decoded, feature=0, anchor_row=5, points=5)
    assert reference.target == neural
    assert reference.anchor_inputs == tuple(float(v) for v in second.after.inputs[5])
    assert json.loads(reference.environment_json) == raw


@pytest.mark.parametrize("change", ["target", "scales", "rows", "version", "depth", "budget"])
def test_link_metadata_rejects_before_any_covariance_replay(
    monkeypatch: pytest.MonkeyPatch, change: str
):
    base, _ = environment("regression", "fresh")
    linked = p.generate_gp_link_pair(base, base.comparison.replacement)
    raw = json.loads(linked.after.recipe_json)["recipe"]
    limits = p.CausalLimits()
    if change == "target":
        raw["target"]["weights"].append([0.3])
    if change == "scales":
        raw["scales"] = [0.0]
    if change == "rows":
        raw["scores"].pop()
    if change == "version":
        raw["parent"]["version"] = "unknown"
    if change == "depth":
        layer = {k: v for k, v in raw.items() if k != "parent"}
        for _ in range(8):
            raw = {**layer, "parent": raw}
    if change == "budget":
        limits = p.CausalLimits(working_bytes=1024)
    calls: list[int] = []

    def forbidden(*args: Any, **kwargs: Any):
        calls.append(1)
        raise AssertionError("replay before complete lineage admission")

    monkeypatch.setattr(np.linalg, "cholesky", forbidden)
    with pytest.raises(ValueError, match="GP|limit|archive|field"):
        p.gp_column_target_link_from_dict(raw, limits=limits)
    assert not calls


def test_link_frozen_receipt_tamper_and_query_preadmission(monkeypatch: pytest.MonkeyPatch):
    base, _ = environment("regression", "fresh")
    pair = p.generate_gp_link_pair(base, base.comparison.replacement)
    raw = json.loads(pair.after.recipe_json)["recipe"]
    raw["scores"][0][0] += 0.1
    with pytest.raises(ValueError, match="replay"):
        p.gp_column_target_link_from_dict(raw)
    raw = json.loads(pair.after.recipe_json)["recipe"]
    calls: list[str] = []

    def forbidden(*args: Any, **kwargs: Any):
        calls.append("allocation")
        raise AssertionError("allocation before query admission")

    monkeypatch.setattr(np.linalg, "cholesky", forbidden)
    monkeypatch.setattr(np, "linspace", forbidden)
    variations: tuple[dict[str, Any], ...] = (
        {"feature": True},
        {"points": 513},
        {"bounds": (2.0, -2.0)},
        {"limits": p.CausalLimits(working_bytes=1024)},
    )
    for options in variations:
        arguments: dict[str, Any] = {"feature": 0, **options}
        with pytest.raises(ValueError, match="GP|limit|archive"):
            p.gp_environment_reference_from_dict(raw, **arguments)
    assert not calls
