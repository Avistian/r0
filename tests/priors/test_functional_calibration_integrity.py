"""Calibrated graphs reject malformed records and retain extreme-scale statistics."""

import json
from dataclasses import asdict, replace
from typing import Any

import pytest
import torch

from r0.modules import priors


def _recipe() -> priors.CalibratedFunctionalCausalRecipe:
    return priors.CalibratedFunctionalCausalRecipe(
        priors.LinearRecipe(
            rows=16, context_rows=8, features=1, signal_features=1, active_features=1
        ),
        (
            priors.FunctionalCausalNode(
                (),
                priors.FrozenLinearFunction(((1.0,), (2.0,))),
                root=priors.JointRootBase("uniform", 1),
            ),
        ),
        ((0, 0),),
        ((0, 1),),
        calibration_rows=32,
    )


@pytest.mark.parametrize("record", [None, [], [1], "wrong", 42, True])
def test_functional_decoder_rejects_nonobject_records(record: Any) -> None:
    with pytest.raises(ValueError, match="records must be objects"):
        priors.functional_recipe_from_dict(record)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("rows", True),
        ("rows", 33),
        ("center", []),
        ("center", [float("inf")]),
        ("scale", [0.0]),
        ("scale", [-1.0]),
        ("scale", [float("nan")]),
        ("constant_channels", [0, 0]),
        ("constant_channels", [1]),
        ("sample_digest", "wrong"),
        ("prefix_fingerprint", "0" * 64),
        ("stream", "emitted-rows"),
        ("version", "unknown"),
    ],
)
def test_inspection_rejects_corrupted_calibration_records(field: str, value: Any) -> None:
    task = priors.generate_task(_recipe(), seed=28, task_id=0)
    card = json.loads(task.mechanism_json)
    card["calibration"][field] = value
    with pytest.raises(ValueError, match="calibration"):
        priors.explain_generation(replace(task, mechanism_json=json.dumps(card)), [0, 15])


@pytest.mark.parametrize("scale", [1e300, 1e-300])
def test_independent_normalization_cancels_finite_extreme_linear_scale(scale: float) -> None:
    recipe = _recipe()
    scaled = replace(
        recipe,
        nodes=(
            replace(
                recipe.nodes[0],
                function=priors.FrozenLinearFunction(((1.0,), (scale,))),
            ),
        ),
    )
    ordinary = priors.generate_task(recipe, seed=28, task_id=0)
    extreme = priors.generate_task(scaled, seed=28, task_id=0)
    torch.testing.assert_close(extreme.scores, ordinary.scores, rtol=1e-12, atol=1e-12)
    trace = priors.explain_generation(extreme, [15, 0, 15])
    assert trace["signal_max_error"] == trace["target_max_error"] == 0


def test_calibration_row_budget_is_admitted_before_emission() -> None:
    recipe = _recipe()
    required = recipe.admit()["estimated_working_bytes"]
    limited = replace(recipe, limits=replace(recipe.limits, working_bytes=required))
    with pytest.raises(ValueError, match="working memory"):
        replace(limited, calibration_rows=65536)
    decoded = priors.functional_recipe_from_dict(json.loads(json.dumps(asdict(limited))))
    assert decoded == limited


@pytest.mark.parametrize("calibration", [None, [], {"unexpected": 1}])
def test_inspection_rejects_missing_or_malformed_calibration(calibration: Any) -> None:
    task = priors.generate_task(_recipe(), seed=28, task_id=0)
    card = json.loads(task.mechanism_json)
    if calibration is None:
        del card["calibration"]
    else:
        card["calibration"] = calibration
    with pytest.raises(ValueError, match="calibration"):
        priors.explain_generation(replace(task, mechanism_json=json.dumps(card)), [0, 15])
