"""Saved analytic mechanisms can be inspected on the same or new observations."""

import json
import math
from dataclasses import replace

import numpy as np
import pytest
import torch

from r0.modules.episodes import TaskType
from r0.modules.priors import (
    FourierRecipe,
    GaussianProcessRecipe,
    GroupedLinearRecipe,
    LinearRecipe,
    PolynomialRecipe,
    QuadraticRecipe,
    Recipe,
    SmoothRecipe,
    evaluate_signal,
    generate_task,
)


@pytest.mark.parametrize(
    "family",
    ["linear", "grouped", "polynomial", "quadratic", "rff", "periodic", "radial", "sinusoidal"],
)
@pytest.mark.parametrize(
    ("task_type", "classes"), [("regression", 0), ("binary", 2), ("multiclass", 3)]
)
def test_saved_mechanisms_evaluate_their_observed_inputs(
    family: str, task_type: TaskType, classes: int
) -> None:
    controls = LinearRecipe(rows=16, context_rows=8, task_type=task_type, classes=classes)
    recipes: dict[str, Recipe] = {
        "linear": controls,
        "grouped": GroupedLinearRecipe(task=controls, calibration_rows=256),
        "polynomial": PolynomialRecipe(task=controls, calibration_rows=256),
        "quadratic": QuadraticRecipe(task=controls, rank=4, calibration_rows=256),
        "rff": FourierRecipe(task=controls, terms=8, calibration_rows=256),
        "periodic": FourierRecipe(task=controls, family="periodic", terms=8, calibration_rows=256),
        "radial": SmoothRecipe(task=controls, projection="ridge", terms=8, calibration_rows=256),
        "sinusoidal": SmoothRecipe(
            task=controls, family="sinusoidal", terms=8, calibration_rows=256
        ),
    }
    task = generate_task(recipes[family], seed=27, task_id=3)
    actual = evaluate_signal(task, task.inputs)
    torch.testing.assert_close(actual, task.scores, rtol=0, atol=0)
    # Reordering and selecting observations cannot resample the function or fit new statistics.
    indices = torch.tensor([7, 1, 7, 12])
    torch.testing.assert_close(
        evaluate_signal(task, task.inputs[indices]), task.scores[indices], rtol=1e-12, atol=1e-12
    )


def test_new_linear_grid_uses_the_saved_function_without_observation_noise() -> None:
    task = generate_task(LinearRecipe(noise_ratio=3), seed=27, task_id=0)
    x = torch.tensor([[float(i + j) / 7 for j in range(8)] for i in range(19)], dtype=torch.float64)
    weights = json.loads(task.mechanism_json)["weights"]
    expected = [math.fsum(float(v) * w[0] for v, w in zip(row, weights, strict=True)) for row in x]
    np.testing.assert_allclose(
        evaluate_signal(task, x).numpy()[:, 0], expected, rtol=1e-12, atol=1e-12
    )
    altered = replace(task, targets=torch.full_like(task.targets, math.nan))
    torch.testing.assert_close(
        evaluate_signal(altered, x), evaluate_signal(task, x), rtol=0, atol=0
    )


def test_exact_gp_does_not_invent_values_outside_its_sampled_support() -> None:
    task = generate_task(GaussianProcessRecipe(calibration_rows=32), seed=27, task_id=0)
    with pytest.raises(ValueError, match="conditional"):
        evaluate_signal(task, task.inputs[:1])


def test_inspection_rejects_invalid_grids_and_nonfinite_results() -> None:
    task = generate_task(LinearRecipe(), seed=27, task_id=0)
    for grid in (
        torch.zeros((0, 8), dtype=torch.float64),
        torch.zeros((20001, 8), dtype=torch.float64),
        torch.zeros((2, 7), dtype=torch.float64),
        torch.zeros((2, 8), dtype=torch.float32),
        torch.full((2, 8), math.nan, dtype=torch.float64),
    ):
        with pytest.raises(ValueError, match="inputs must"):
            evaluate_signal(task, grid)
    polynomial = generate_task(PolynomialRecipe(calibration_rows=256), seed=27, task_id=0)
    with pytest.raises(ValueError, match="nonfinite evaluated signal"):
        evaluate_signal(polynomial, torch.full((2, 8), 1e200, dtype=torch.float64))
