"""Fitted normalization and range warps use the ordinary saved explorer path."""

import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pytest

from r0.modules import priors
from r0.modules.episodes import TaskType
from r0.modules.explorer import ExplorerSession


@pytest.mark.parametrize("kind", ["l2_normalize", "kumaraswamy"])
@pytest.mark.parametrize("task_type", ["regression", "binary", "multiclass"])
def test_fitted_transform_saves_reopens_and_traces(
    tmp_path: Path, kind: str, task_type: TaskType
) -> None:
    weights = ((1.0, 0.5, -1.0), (-0.25, 1.0, 2.0))
    sample = np.random.default_rng(128).normal(size=(32, 2)) @ np.asarray(weights)
    transform = (
        priors.calibrate_l2_normalization(sample, calibration_id="independent-parent-fixture")
        if kind == "l2_normalize"
        else priors.calibrate_kumaraswamy(
            sample, a=0.7, b=1.3, calibration_id="independent-parent-fixture"
        )
    )
    controls = priors.LinearRecipe(
        rows=16,
        context_rows=8,
        features=2,
        signal_features=2,
        active_features=2,
        task_type=task_type,
        classes={"regression": 0, "binary": 2, "multiclass": 3}[task_type],
    )
    recipe = priors.CausalRecipe(
        task=controls,
        nodes=(
            priors.CausalNode(bias=(0.0, 0.0), noise_scale=(1.0, 1.0)),
            priors.ActivatedCausalNode(
                parents=(0,),
                weights=weights,
                bias=(0.0, 0.0, 0.0),
                noise_scale=(0.0, 0.0, 0.0),
                output_activation=transform,
            ),
        ),
        observed=((0, 0), (0, 1)),
        target=tuple((1, i) for i in range(3 if task_type == "multiclass" else 1)),
        version="causal-v2",
    )
    session = ExplorerSession(tmp_path / "original")
    result = session.request(
        "generate", {"record": {"recipe": asdict(recipe), "seed": 28, "task_id": 0}}
    )
    assert json.loads(result["recipe_json"])["recipe"]["nodes"][1]["output_activation"] == (
        json.loads(json.dumps(asdict(transform)))
    )
    assert session.request("graph", {"indices": [0, 15]})["graph"]["target_max_error"] < 1e-12
    reopened = ExplorerSession(tmp_path / "original").request(
        "open", {"archive": result["archive"], "task_id": 0}
    )
    assert reopened["view"] == result["view"]
    replay = ExplorerSession(tmp_path / "replay")
    repeated = replay.request("generate", {"record_json": result["recipe_json"]})
    assert repeated["recipe_json"] == result["recipe_json"]
    assert replay.request("row", {"index": 15}) == session.request("row", {"index": 15})
