"""Target environments preserve finite-population custody and shared adapter draws."""

import copy
import json
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch
from numpy.typing import NDArray

from r0.modules import generation as g
from r0.modules import priors as p


def original(task_type: Any = "regression") -> p.GPColumnRecipe:
    classes = {"regression": 0, "binary": 2, "multiclass": 3}[task_type]
    pop = p.realize_gp_columns(
        p.GPColumnRequest(rows=16, columns=2, outputs=3 if classes == 3 else 1),
        seed=18014398509482003,
        task_id=18014398509482001,
    )
    return p.GPColumnRecipe(
        pop,
        p.LinearRecipe(
            rows=16,
            context_rows=8,
            features=2,
            signal_features=2,
            active_features=2,
            task_type=task_type,
            classes=classes,
            noise_ratio=0.5 if not classes else 0.0,
        ),
    )


def replacement(family: str, outputs: int) -> p.GPColumnTarget:
    weights = tuple(
        tuple((i + 1) * (-1.0 if k % 2 else 1.0) for k in range(outputs)) for i in range(2)
    )
    return p.GPColumnTarget(
        family,
        weights,
        (0.4,) * outputs,
        quadratic=tuple(((0.3, 0.2), (-0.1, -0.4)) for _ in range(outputs))
        if family == "quadratic"
        else (),
        hidden=((0.2, -0.7), (0.6, 0.5)) if family == "neural" else (),
        hidden_bias=(0.3, -0.2) if family == "neural" else (),
    )


@pytest.mark.parametrize("family", ["linear", "quadratic", "neural"])
@pytest.mark.parametrize("task_type", ["regression", "binary", "multiclass"])
@pytest.mark.parametrize("mode", ["matched", "fresh"])
def test_target_pair_literal_calibration_and_shared_adapter(family: str, task_type: str, mode: str):
    recipe = original(task_type)
    pop = recipe.source
    assert isinstance(pop, p.GPColumnPopulation)
    target = replacement(family, pop.request.outputs)
    pair = p.generate_gp_column_pair(
        recipe, target, mode=mode, fresh_seed=19 if mode == "fresh" else None
    )
    record = json.loads(pair.after.recipe_json)
    comparison = record["recipe"]["comparison"]
    assert comparison["original"] == json.loads(json.dumps(asdict(pop)))
    assert torch.equal(pair.before.inputs, pair.after.inputs)
    assert torch.equal(pair.before.context_indices, pair.after.context_indices)
    x = pair.after.inputs.numpy()
    cal = np.asarray(pop.calibration_inputs)

    def equation(values: NDArray[np.float64]) -> NDArray[np.float64]:
        z = np.tanh(values @ target.hidden + target.hidden_bias) if family == "neural" else values
        raw = z @ target.weights + target.bias
        if family == "quadratic":
            for k, q in enumerate(target.quadratic):
                raw[:, k] += np.sum((values @ q) * values, axis=1)
        return raw

    fit = equation(cal)
    expected = (equation(x) - fit.mean(axis=0)) / fit.std(axis=0)
    np.testing.assert_allclose(pair.after.scores.numpy(), expected, atol=2e-14)
    if mode == "matched":
        baseline = p.generate_task(recipe, seed=pop.seed, task_id=pop.task_id)
        for field in ("inputs", "scores", "targets", "context_indices", "query_indices"):
            assert torch.equal(getattr(pair.before, field), getattr(baseline, field))
    else:
        assert not np.array_equal(x, pop.draw.values)
    if task_type == "regression":
        np.testing.assert_allclose(
            (pair.before.targets - pair.before.scores[:, 0]).numpy(),
            (pair.after.targets - pair.after.scores[:, 0]).numpy(),
            atol=1e-15,
        )
    if task_type != "regression":
        if task_type == "binary":
            positive = 1 / (1 + np.exp(-expected[:, 0]))
            probabilities = np.column_stack((1 - positive, positive))
        else:
            exponentials = np.exp(expected - expected.max(axis=1, keepdims=True))
            probabilities = exponentials / exponentials.sum(axis=1, keepdims=True)
        assert pair.after.probabilities is not None
        np.testing.assert_allclose(pair.after.probabilities.numpy(), probabilities, atol=2e-14)
    assert p.compare_causal_pair(pair.before, pair.after)["kind"] == "gp-target-edit"


def test_target_pair_zip_replay_and_repeated_original_rows(tmp_path: Path):
    recipe = original("multiclass")
    pair = p.generate_gp_column_pair(
        recipe, replacement("neural", 3), mode="fresh", fresh_seed=2**54 + 7
    )
    g.save_causal_pair(tmp_path / "pair", pair, max_bytes=128 * 1024**2)
    loaded = g.load_causal_pair(tmp_path / "pair")
    assert json.loads(loaded.card_json) == json.loads(pair.card_json)
    graph = p.explain_generation(loaded.after, [7, 2, 7])
    assert graph["rows"] == [7, 2, 7]
    inputs = next(n for n in graph["nodes"] if n["id"] == "inputs")
    np.testing.assert_allclose(inputs["values"], pair.after.inputs[[7, 2, 7]].numpy())
    assert inputs["parents"] == ["locations", "innovations"]
    decoded = p.recipe_from_dict(json.loads(pair.after.recipe_json)["recipe"])
    assert isinstance(decoded, p.GPColumnTargetRecipe)
    assert decoded.side == "replacement"


@pytest.mark.parametrize("change", ["side", "task", "version", "mode", "seed", "cap"])
def test_target_environment_metadata_rejects_before_cholesky(
    monkeypatch: pytest.MonkeyPatch, change: str
):
    recipe = original()
    pair = p.generate_gp_column_pair(recipe, replacement("linear", 1))
    raw = copy.deepcopy(json.loads(pair.after.recipe_json)["recipe"])
    limits = p.CausalLimits()
    if change == "side":
        raw["side"] = "unknown"
    if change == "task":
        raw["task"]["features"] = 3
    if change == "version":
        raw["version"] = "wrong"
    if change == "mode":
        raw["comparison"]["mode"] = "unknown"
    if change == "seed":
        raw["comparison"]["fresh_seed"] = True
    if change == "cap":
        limits = replace(limits, working_bytes=1024)
    calls: list[int] = []

    def forbidden(*args: Any, **kwargs: Any):
        calls.append(1)
        raise AssertionError("Cholesky before outer admission")

    monkeypatch.setattr(np.linalg, "cholesky", forbidden)
    with pytest.raises(ValueError, match="GP|archive|limit|geometry|Intervention"):
        p.gp_column_target_recipe_from_dict(raw, limits=limits)
    assert not calls


@pytest.mark.parametrize(
    "field",
    ["task_type", "targets_dtype", "classes", "split", "probabilities", "execution", "identity"],
)
def test_target_pair_rejects_forged_semantics(field: str):
    recipe = original("binary")
    pair = p.generate_gp_column_pair(recipe, replacement("neural", 1))
    changes: dict[str, Any] = {}
    if field == "task_type":
        changes["task_type"] = "regression"
    if field == "targets_dtype":
        changes["targets"] = pair.after.targets.to(torch.float64)
    if field == "classes":
        changes["classes"] = ("wrong", "names")
    if field == "split":
        changes["query_indices"] = pair.after.context_indices
    if field == "probabilities":
        changes["probabilities"] = None
    if field in ("execution", "identity"):
        record = json.loads(pair.after.recipe_json)
        if field == "execution":
            record["execution"]["implementation"] = "wrong"
        else:
            record["task_id"] = float(record["task_id"])
        changes["recipe_json"] = json.dumps(record)
    with pytest.raises(ValueError, match="GP"):
        p.compare_causal_pair(pair.before, replace(pair.after, **changes))


def test_target_pair_historical_environment_and_aggregate_boundary(monkeypatch: pytest.MonkeyPatch):
    recipe = original()
    target = replacement("quadratic", 1)
    pair = p.generate_gp_column_pair(recipe, target)
    record = json.loads(pair.after.recipe_json)
    record["execution"]["python"] = "historical"
    record["execution"]["cpu_threads"] = 17
    assert p.compare_causal_pair(
        pair.before, replace(pair.after, recipe_json=json.dumps(record))
    ) == json.loads(pair.card_json)
    low, high = 1, p.CausalLimits().working_bytes
    while low < high:
        middle = (low + high) // 2
        try:
            p.validate_gp_target_edit_controls(
                asdict(recipe), asdict(target), limits=p.CausalLimits(working_bytes=middle)
            )
        except ValueError:
            low = middle + 1
        else:
            high = middle
    calls: list[int] = []

    def forbidden(*args: Any, **kwargs: Any):
        calls.append(1)
        raise RuntimeError("admitted replay sentinel")

    monkeypatch.setattr(np.linalg, "cholesky", forbidden)
    with pytest.raises(ValueError, match="limit"):
        p.generate_gp_column_pair(recipe, target, limits=p.CausalLimits(working_bytes=low - 1))
    assert not calls
    with pytest.raises(RuntimeError, match="admitted replay sentinel"):
        p.generate_gp_column_pair(recipe, target, limits=p.CausalLimits(working_bytes=low))
    assert calls == [1]


@pytest.mark.parametrize("field", ["original", "replacement_raw", "calibration", "fresh_seed_type"])
def test_target_environment_full_receipt_replay_rejects_tamper(field: str):
    recipe = original()
    pair = p.generate_gp_column_pair(
        recipe, replacement("quadratic", 1), mode="fresh", fresh_seed=17
    )
    raw = json.loads(pair.after.recipe_json)["recipe"]
    c = raw["comparison"]
    if field == "original":
        c["original"]["draw"]["innovations"][0][0] += 0.1
    if field == "replacement_raw":
        c["replacement_raw"][0][0] += 0.1
    if field == "calibration":
        c["scales"][0] *= 2
    if field == "fresh_seed_type":
        c["fresh_seed"] = 17.0
    with pytest.raises(ValueError, match="GP"):
        p.gp_column_target_recipe_from_dict(raw)


def test_constant_target_keeps_explicit_unit_fallback_without_query_fit():
    recipe = original()
    target = p.GPColumnTarget("linear", ((0.0,), (0.0,)), (3.0,))
    pair = p.generate_gp_column_pair(recipe, target)
    retained = json.loads(pair.after.recipe_json)["recipe"]["comparison"]
    assert retained["centers"] == [3.0]
    assert retained["scales"] == [1.0]
    assert torch.equal(pair.after.scores, torch.zeros_like(pair.after.scores))
