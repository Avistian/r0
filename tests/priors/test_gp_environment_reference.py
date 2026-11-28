"""Every comparison side evaluates its own equation, calibration and actual row anchor."""

import json
from dataclasses import asdict
from typing import Any

import numpy as np
import pytest

from r0.modules import priors as p


@pytest.mark.parametrize("family", ["linear", "quadratic", "neural"])
@pytest.mark.parametrize("kind", ["regression", "binary", "multiclass"])
@pytest.mark.parametrize("mode", ["matched", "fresh"])
def test_environment_reference_literal_selected_side(family: str, kind: Any, mode: str):
    classes = {"regression": 0, "binary": 2, "multiclass": 3}[kind]
    pop = p.realize_gp_columns(
        p.GPColumnRequest(rows=16, target_family=family, outputs=3 if classes == 3 else 1),
        seed=13,
        task_id=2**54 + 7,
    )
    task = p.LinearRecipe(
        rows=16,
        context_rows=8,
        features=2,
        signal_features=2,
        active_features=2,
        task_type=kind,
        classes=classes,
        noise_ratio=0.7 if not classes else 0.0,
    )
    target = p.GPColumnTarget(
        "linear",
        tuple(tuple((i + 1) * (k + 1) * 0.3 for k in range(pop.request.outputs)) for i in range(2)),
        (0.1,) * pop.request.outputs,
    )
    pair = p.generate_gp_column_pair(
        p.GPColumnRecipe(pop, task),
        target,
        mode=mode,
        fresh_seed=2**54 + 9 if mode == "fresh" else None,
    )
    for selected in (pair.before, pair.after):
        recipe = p.recipe_from_dict(json.loads(selected.recipe_json)["recipe"])
        assert isinstance(recipe, p.GPColumnTargetRecipe)
        result = p.evaluate_gp_environment_reference(
            recipe, feature=1, anchor_row=3, points=7, bounds=(-2.0, 3.0)
        )
        assert json.loads(result.environment_json) == json.loads(selected.recipe_json)["recipe"]
        assert result.anchor_inputs == tuple(float(v) for v in selected.inputs[3])
        assert result.mode == mode
        assert result.fresh_seed == (2**54 + 9 if mode == "fresh" else None)
        equation = result.target
        x = np.repeat(selected.inputs[3:4].numpy(), 7, axis=0)
        x[:, 1] = np.linspace(-2.0, 3.0, 7)
        h = (
            np.tanh(x @ equation.hidden + equation.hidden_bias)
            if equation.family == "neural"
            else x
        )
        raw = h @ equation.weights + equation.bias
        if equation.family == "quadratic":
            for k, q in enumerate(equation.quadratic):
                raw[:, k] += np.sum((x @ q) * x, axis=1)
        z = (raw - result.centers) / result.scales
        np.testing.assert_allclose(result.raw_scores, raw, atol=1e-13)
        np.testing.assert_allclose(result.scores, z, atol=1e-13)
        if classes:
            if classes == 2:
                positive = 1 / (1 + np.exp(-z[:, 0]))
                wanted = np.column_stack((1 - positive, positive))
            else:
                exponentials = np.exp(z - z.max(axis=1, keepdims=True))
                wanted = exponentials / exponentials.sum(axis=1, keepdims=True)
            assert result.probabilities is not None
            np.testing.assert_allclose(result.probabilities, wanted, atol=1e-13)
        else:
            assert result.regression_mean is not None
            np.testing.assert_allclose(result.regression_mean, z[:, 0], atol=1e-13)
            assert result.observation_noise_std == 0.7
        assert asdict(recipe) == json.loads(result.environment_json) or json.loads(
            json.dumps(asdict(recipe))
        ) == json.loads(result.environment_json)
