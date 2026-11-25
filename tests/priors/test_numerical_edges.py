"""Inactive features and representable weak variation must not destroy valid signals."""

import json
import math
from dataclasses import replace

import pytest
import torch

from r0.modules.priors import (
    FourierRecipe,
    InputDistribution,
    LinearRecipe,
    PolynomialRecipe,
    evaluate_signal,
    explain_generation,
    generate_task,
)


@pytest.mark.parametrize("family", ["rff", "periodic"])
@pytest.mark.parametrize("length_scale", [1e13, 1e200])
def test_nearly_flat_fourier_function_keeps_its_normalized_shape(
    family: str, length_scale: float
) -> None:
    recipe = FourierRecipe(
        task=LinearRecipe(features=1, signal_features=1, active_features=1),
        family=family,
        terms=1,
        calibration_rows=256,
        length_scale=1e7,
    )
    reference = generate_task(recipe, seed=27, task_id=0)
    weak = generate_task(replace(recipe, length_scale=length_scale), seed=27, task_id=0)
    torch.testing.assert_close(weak.scores, reference.scores, rtol=2e-6, atol=2e-6)
    card = json.loads(weak.mechanism_json)
    assert card["centered_basis"] is True
    projection = float(weak.inputs[0, 0]) * card["frequencies"][0][0]
    raw = -2 * math.sin(projection / 2) * math.sin(card["phases"][0] + projection / 2)
    expected = (raw * card["coefficients"][0][0] - card["center"][0]) / card["scale"][0]
    assert float(weak.scores[0, 0]) == pytest.approx(expected, rel=1e-12, abs=1e-12)
    graph = explain_generation(weak, [0, 1])
    assert graph["signal_max_error"] == graph["target_max_error"] == 0
    json.dumps(graph, allow_nan=False)
    assert (
        weak.episode()
        == generate_task(replace(recipe, length_scale=length_scale), seed=27, task_id=0).episode()
    )


def test_polynomial_signal_ignores_huge_inactive_inputs() -> None:
    task = generate_task(PolynomialRecipe(degree=16, calibration_rows=256), seed=27, task_id=0)
    active = json.loads(task.mechanism_json)["active"]
    inputs = task.inputs.clone()
    inactive = [i for i in range(inputs.shape[1]) if i not in active]
    inputs[:, inactive] = 1e200
    assert torch.equal(evaluate_signal(task, inputs), task.scores)


def test_inactive_polynomial_powers_do_not_poison_calibration_or_graph() -> None:
    recipe = PolynomialRecipe(
        task=LinearRecipe(features=128, signal_features=128, active_features=1),
        inputs=InputDistribution(scale=5e18),
        degree=16,
        calibration_rows=256,
    )
    task = generate_task(recipe, seed=1, task_id=0)
    card = json.loads(task.mechanism_json)
    assert card["active"] == [29]
    raw = float(task.inputs[0, 29]) ** 16 * card["weights"][29][0]
    expected = (raw - card["center"][0]) / card["scale"][0]
    assert float(task.scores[0, 0]) == pytest.approx(expected, rel=1e-12)
    graph = explain_generation(task, [0, 1])
    basis = next(node for node in graph["nodes"] if node["id"] == "basis")
    assert "zero for inactive" in basis["formula"]
    assert all(row[i] == 0 for row in basis["values"] for i in range(32) if i != 29)
    assert graph["signal_max_error"] == graph["target_max_error"] == 0
    json.dumps(graph, allow_nan=False)
