"""Malformed fit-inspection transport rejects before covariance reconstruction."""

import json
from dataclasses import replace
from typing import Any

import numpy as np
import pytest

from r0.modules import priors as p


@pytest.fixture(scope="module")
def task():
    source = p.realize_gp_columns(p.GPColumnRequest(rows=16), seed=17, task_id=19)
    controls = p.LinearRecipe(
        rows=16, context_rows=8, features=2, signal_features=2, active_features=2
    )
    root = p.FunctionalCausalNode(
        (), p.FrozenLinearFunction(((1.0, 0.0), (0.0, 1.0))), p.JointRootBase("gaussian", 2)
    )
    child = p.FunctionalCausalNode((0,), p.FrozenLinearFunction(((2.0, -1.0),)))
    base = p.FunctionalCausalRecipe(controls, (root, child), ((0, 0), (0, 1)), ((1, 0),))
    fit = p.calibrate_gp_root_dag(
        p.GPRootDAGRecipe(source, base, (p.GPRootBinding(0, (0, 1)),)),
        calibration_rows=32,
        calibration_seed=23,
        calibration_task_id=29,
    )
    return p.generate_task(fit, seed=17, task_id=19)


@pytest.mark.parametrize(
    "payload", [{}, None, [], 3, "wrong", {"recipe": None}, {"recipe": []}, {"recipe": 4}]
)
def test_malformed_fit_transport_is_value_error_before_replay(
    task: p.TrainingTask, payload: Any, monkeypatch: pytest.MonkeyPatch
):
    def forbidden(*args: Any, **kwargs: Any):
        pytest.fail("malformed transport reached covariance reconstruction")

    monkeypatch.setattr(np.linalg, "cholesky", forbidden)
    with pytest.raises(
        ValueError, match="malformed GP root task transport|Expecting property name"
    ):
        p.explain_gp_calibration_fit(replace(task, recipe_json=json.dumps(payload)), [31, 0, 31])


@pytest.mark.parametrize("field", ["recipe", "seed", "task_id", "execution", "extra"])
def test_fit_transport_requires_complete_outer_schema(
    task: p.TrainingTask, field: str, monkeypatch: pytest.MonkeyPatch
):
    raw = json.loads(task.recipe_json)
    if field == "extra":
        raw[field] = None
    else:
        del raw[field]

    def forbidden(*args: Any, **kwargs: Any):
        pytest.fail("replay before schema")

    monkeypatch.setattr(np.linalg, "cholesky", forbidden)
    with pytest.raises(
        ValueError, match="malformed GP root task transport|Expecting property name"
    ):
        p.explain_gp_calibration_fit(replace(task, recipe_json=json.dumps(raw)), [0])


@pytest.mark.parametrize("raw", ["{", "null", "[]"])
def test_invalid_json_fit_transport_is_value_error(task: p.TrainingTask, raw: str):
    with pytest.raises(
        ValueError, match="malformed GP root task transport|Expecting property name"
    ):
        p.explain_gp_calibration_fit(replace(task, recipe_json=raw), [0])
