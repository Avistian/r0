"""Guarded fitted-graph curves use explicit causal references and retained models."""

import importlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from r0.modules.explorer import ExplorerSession


@pytest.mark.parametrize("objective", ["reg:squarederror", "binary:logistic", "multi:softprob"])
def test_fitted_curve_matches_native_model_and_preserves_archive(tmp_path: Path, objective: str):
    session = ExplorerSession(tmp_path)
    controls = session.request("construction_examples", {})["examples"][
        f"XGBoost shared-vector fit: {objective}"
    ]
    controls["skeleton"]["task"].update(rows=16, context_rows=8)
    controls["fits"][0]["plan"].update(rows=32, estimators=2)
    created = session.request(
        "construct",
        {
            "controls": controls,
            "construction_seed": "31",
            "seed": "17",
            "task_id": "18014398509481987",
        },
    )
    original = created["recipe_json"]
    curve = session.request("curve", {"feature": 0, "anchor_row": 3, "points": 9})["curve"]
    assert curve["kind"] == "functional_causal_reference"
    assert curve["clamp_location"] == "state_and_readout"
    assert curve["reference"] == "original-row"
    assert curve["original_task_id"] == "18014398509481987"
    raw = json.loads(original)["recipe"]
    xgb: Any = importlib.import_module("xgboost")
    model = xgb.Booster()
    model.load_model(bytearray(raw["fits"][0]["realization"]["model_json"].encode()))
    inputs = np.repeat(np.array(curve["anchor_inputs"])[None, :], 9, axis=0)
    inputs[:, 0] = curve["values"]
    expected = model.predict(xgb.DMatrix(inputs))[:, :1]
    np.testing.assert_allclose(
        np.asarray(curve["scores"], dtype=np.float64), expected, rtol=0, atol=1e-7
    )
    zero = session.request(
        "curve", {"feature": 0, "anchor_row": 3, "points": 9, "reference": "zero"}
    )["curve"]
    np.testing.assert_array_equal(zero["scores"], curve["scores"])
    assert session.task is not None
    assert session.task.recipe_json == original
    with pytest.raises(ValueError, match="curve failed"):
        session.request("curve", {"feature": 0, "points": 9, "causal_limits": {"working_bytes": 1}})
    with pytest.raises(ValueError, match="timed out"):
        session.request("curve", {"feature": 0, "points": 9, "max_seconds": 0.001})
    assert session.task.recipe_json == original
