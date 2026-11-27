"""Frozen GP target slices use original calibration and shared class adapters."""

import copy
from dataclasses import asdict
from typing import Any

import numpy as np
import pytest

from r0.modules import priors as p


@pytest.mark.parametrize("family", ["linear", "quadratic", "neural"])
@pytest.mark.parametrize(
    ("task_type", "classes"), [("regression", 0), ("binary", 2), ("multiclass", 3)]
)
def test_gp_target_reference_literal_equations_and_adapters(
    family: str, task_type: Any, classes: int
):
    request = p.GPColumnRequest(
        rows=16, columns=2, target_family=family, outputs=3 if classes == 3 else 1
    )
    population = p.realize_gp_columns(request, seed=29, task_id=18014398509482001)
    task = p.LinearRecipe(
        rows=16,
        context_rows=8,
        features=2,
        signal_features=2,
        active_features=2,
        task_type=task_type,
        classes=classes,
        noise_ratio=0.7 if not classes else 0.0,
        separation=1.4 if classes else 1.0,
        class_offsets=(0.3, -0.5, 0.7) if classes == 3 else (0.3,) if classes else (),
    )
    recipe = p.GPColumnRecipe(population, task)
    original = asdict(recipe)
    result = p.evaluate_gp_column_reference(
        recipe, feature=1, anchor_row=3, points=7, bounds=(-1.7, 2.3)
    )
    x = np.repeat(np.asarray(population.draw.values)[3:4], 7, axis=0)
    x[:, 1] = np.linspace(-1.7, 2.3, 7)
    target = population.target
    inputs = (
        np.tanh(x @ np.asarray(target.hidden) + target.hidden_bias) if family == "neural" else x
    )
    raw = inputs @ np.asarray(target.weights) + target.bias
    if family == "quadratic":
        for k, matrix in enumerate(target.quadratic):
            for row in range(7):
                raw[row, k] += sum(
                    x[row, i] * matrix[i][j] * x[row, j] for i in range(2) for j in range(2)
                )
    normalized = (raw - population.centers) / population.scales
    np.testing.assert_allclose(result.raw_scores, raw, rtol=1e-13, atol=1e-13)
    np.testing.assert_allclose(result.scores, normalized, rtol=1e-13, atol=1e-13)
    if classes:
        logits = 1.4 * normalized + np.asarray(task.class_offsets)
        if classes == 2:
            positive = 1 / (1 + np.exp(-logits[:, 0]))
            expected = np.column_stack((1 - positive, positive))
        else:
            unnormalized = np.exp(logits - logits.max(axis=1, keepdims=True))
            expected = unnormalized / unnormalized.sum(axis=1, keepdims=True)
        assert result.probabilities is not None
        np.testing.assert_allclose(result.probabilities, expected, rtol=1e-13, atol=1e-13)
        assert result.regression_mean is None
        assert result.observation_noise_std is None
    else:
        assert result.regression_mean is not None
        np.testing.assert_allclose(result.regression_mean, normalized[:, 0])
        assert result.observation_noise_std == 0.7
        assert result.probabilities is None
    assert result.population_id == population.population_id
    assert result.seed == 29
    assert result.task_id == 18014398509482001
    assert result.centers == population.centers
    assert result.scales == population.scales
    assert asdict(recipe) == original


@pytest.mark.parametrize("change", ["feature", "row", "points", "bounds", "budget"])
def test_gp_reference_controls_and_query_budget_precede_covariance_and_grid(
    monkeypatch: pytest.MonkeyPatch, change: str
):
    population = p.realize_gp_columns(p.GPColumnRequest(rows=16), seed=11, task_id=13)
    recipe = p.GPColumnRecipe(
        population,
        p.LinearRecipe(rows=16, context_rows=8, features=2, signal_features=2, active_features=2),
    )
    controls: dict[str, Any] = {"feature": 0, "points": 512}
    controls.update(
        {
            "feature": {"feature": True},
            "row": {"anchor_row": 16},
            "points": {"points": 513},
            "bounds": {"bounds": (2.0, -2.0)},
            "budget": {"limits": p.CausalLimits(working_bytes=1024**2)},
        }[change]
    )

    def forbidden(*args: Any, **kwargs: Any):
        raise AssertionError("reference allocated covariance/grid before admission")

    monkeypatch.setattr(copy, "deepcopy", forbidden)
    monkeypatch.setattr(np.linalg, "cholesky", forbidden)
    monkeypatch.setattr(np, "repeat", forbidden)
    with pytest.raises(ValueError, match="reference|limit"):
        p.evaluate_gp_column_reference(recipe, **controls)


def test_gp_reference_fixed_query_preserves_calibration_and_rejects_tamper():
    population = p.realize_gp_columns(
        p.GPColumnRequest(rows=16, target_family="quadratic"), seed=11, task_id=13
    )
    recipe = p.GPColumnRecipe(
        population,
        p.LinearRecipe(rows=16, context_rows=8, features=2, signal_features=2, active_features=2),
    )
    value = population.draw.values[3][0]
    result = p.evaluate_gp_column_reference(
        recipe, feature=0, anchor_row=3, points=5, bounds=(value, value)
    )
    np.testing.assert_allclose(
        result.scores,
        np.repeat(np.asarray(population.scores)[3:4], 5, axis=0),
        rtol=1e-13,
        atol=1e-13,
    )
    data = copy.deepcopy(asdict(recipe))
    changed = [list(row) for row in data["source"]["draw"]["values"]]
    changed[0][0] += 0.01
    data["source"]["draw"]["values"] = changed
    with pytest.raises(ValueError, match="replay|differs"):
        p.gp_column_reference_from_dict(data, feature=0)
