"""Conditional inspection checked against a separate dense linear solve."""

import json
from typing import Any, cast

import numpy as np
import pytest
import torch

from r0.modules.priors import (
    GaussianProcessRecipe,
    LinearRecipe,
    StationaryKernel,
    condition_gp,
    explain_generation,
    generate_task,
)


def test_conditioning_matches_dense_matern_solve_without_changing_task() -> None:
    task = generate_task(
        GaussianProcessRecipe(
            task=LinearRecipe(rows=16, context_rows=8),
            calibration_rows=32,
            kernels=(StationaryKernel(kind="matern12", length_scale=0.7),),
        ),
        seed=27,
        task_id=4,
    )
    before = task.episode(), task.mechanism_json
    card = json.loads(task.mechanism_json)
    query = torch.full((3, task.inputs.shape[1]), 0.3, dtype=torch.float64)
    query[:, card["active"][0]] = torch.tensor([-0.4, 0.6, 1.2])
    result = condition_gp(task, query)
    support = np.array(card["support"])
    points = query.numpy()[:, card["active"]]

    def kernel(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        distance = np.sqrt(((a[:, None] - b[None, :]) ** 2).sum(2))
        return np.exp(-distance / 0.7)

    # All saved support values are reconstructed from the original symmetric root.
    k = kernel(support, support)
    eigenvalues, vectors = np.linalg.eigh(k)
    original = (
        (vectors * np.sqrt(np.maximum(eigenvalues, 0)))
        @ vectors.T
        @ np.array(card["standard_normal"])
    )
    cross = kernel(points, support)
    expected_mean = (cross @ np.linalg.solve(k, original) - card["center"]) / card["scale"]
    expected_covariance = kernel(points, points) - cross @ np.linalg.solve(k, cross.T)
    np.testing.assert_allclose(result["mean"], expected_mean, rtol=1e-9, atol=1e-10)
    np.testing.assert_allclose(result["raw_covariance"], expected_covariance, atol=1e-10)
    assert result["conditioning"]["retained_rank"] == len(support)
    assert result["kind"] == "gp_conditional"
    assert result["draws"] == []
    assert result["conditioning"]["observation_noise"] == 0
    assert before == (task.episode(), task.mechanism_json)
    graph = explain_generation(task, [0, 3])
    assert graph["target_max_error"] < 1e-10
    json.dumps(result, allow_nan=False)


@pytest.mark.parametrize("classes", [0, 2, 3])
def test_additional_draws_preserve_saved_values_and_repeat_query_points(classes: int) -> None:
    task = generate_task(
        GaussianProcessRecipe(
            task=LinearRecipe(
                rows=16,
                context_rows=8,
                features=6,
                signal_features=4,
                task_type="regression"
                if classes == 0
                else "binary"
                if classes == 2
                else "multiclass",
                classes=classes,
            ),
            calibration_rows=32,
            kernels=(StationaryKernel(kind="matern12"),),
        ),
        seed=27,
        task_id=1,
    )
    query = torch.stack(
        (task.inputs[0], torch.zeros_like(task.inputs[0]), torch.zeros_like(task.inputs[0]))
    )
    query[2, 5] = 4  # An irrelevant column cannot change the function.
    result = condition_gp(task, query, draws=3, seed=91)
    again = condition_gp(task, query, draws=3, seed=91)
    assert result == again
    values = np.array(result["draws"])
    np.testing.assert_allclose(
        values[:, 0], np.repeat(task.scores[0:1].numpy(), 3, axis=0), atol=1e-10
    )
    np.testing.assert_array_equal(values[:, 1], values[:, 2])
    assert np.std(values[:, 1]) > 0
    assert np.array(result["standard_deviation"])[0, 0] == 0
    other = condition_gp(task, query, draws=3, seed=92)
    assert result["mean"] == other["mean"]
    assert result["draws"] != other["draws"]
    reordered = condition_gp(task, query[[2, 0, 1]], draws=3, seed=91)
    np.testing.assert_array_equal(np.array(reordered["draws"])[:, [1, 2, 0]], values)
    assert result["sampling"]["seed"] == 91
    assert (
        result["sampling"]["query_set_fingerprint"]
        == reordered["sampling"]["query_set_fingerprint"]
    )


def test_rank_truncation_is_explicit_and_reports_changed_support_values() -> None:
    task = generate_task(
        GaussianProcessRecipe(
            task=LinearRecipe(
                rows=16, context_rows=8, features=1, signal_features=1, active_features=1
            ),
            calibration_rows=32,
            kernels=(StationaryKernel(length_scale=2),),
        ),
        seed=27,
        task_id=8,
    )
    query = torch.tensor([[-0.5], [0.1], [0.7]], dtype=torch.float64)
    with pytest.raises(ValueError, match="rank deficient"):
        condition_gp(task, query)
    result = condition_gp(task, query, draws=2, seed=42, allow_rank_truncation=True)
    diagnostics = result["conditioning"]
    assert diagnostics["approximate"] is True
    assert 0 < diagnostics["retained_rank"] < diagnostics["support_count"]
    assert diagnostics["discarded_support_max_error"] > 0
    assert np.isfinite(result["draws"]).all()
    saved = condition_gp(task, task.inputs[:2], draws=2, seed=42, allow_rank_truncation=True)
    np.testing.assert_allclose(saved["mean"], task.scores[:2].numpy(), atol=1e-10)
    np.testing.assert_allclose(saved["draws"][0], task.scores[:2].numpy(), atol=1e-10)
    assert np.array(saved["standard_deviation"]).max() == 0


def test_conditional_inspection_rejects_ambiguous_sampling_and_invalid_inputs() -> None:
    task = generate_task(
        GaussianProcessRecipe(kernels=(StationaryKernel(kind="matern12"),)), seed=2, task_id=3
    )
    query = task.inputs[:2]
    for kwargs in (
        {"draws": -1},
        {"draws": True},
        {"draws": 9},
        {"draws": 1},
        {"seed": 0},
        {"draws": 1, "seed": -1},
        {"allow_rank_truncation": 1},
    ):
        with pytest.raises(ValueError, match="draws|seed|boolean"):
            condition_gp(task, query, **cast(Any, kwargs))
    for invalid in (
        query.float(),
        query[:, :1],
        query[:0],
        torch.full_like(query, float("nan")),
        query.repeat(129, 1),
    ):
        with pytest.raises(ValueError, match="inputs"):
            condition_gp(task, invalid)
    linear = generate_task(LinearRecipe(), seed=2, task_id=3)
    with pytest.raises(ValueError, match="exact GP"):
        condition_gp(linear, linear.inputs[:2])
