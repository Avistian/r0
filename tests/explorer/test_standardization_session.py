"""Standalone source Standardize survives saved task-type workflows without refitting."""

from dataclasses import asdict
from pathlib import Path

import numpy as np
import pytest

from r0.modules import priors
from r0.modules.episodes import TaskType
from r0.modules.explorer import ExplorerSession


@pytest.mark.parametrize("task_type", ["regression", "binary", "multiclass"])
def test_standardization_saves_reopens_and_replays(tmp_path: Path, task_type: TaskType) -> None:
    fitted = priors.calibrate_standardization(
        np.array([[1.0, -2.0, 4.0], [3.0, 2.0, 4.0]]),
        calibration_id="saved-independent-standardize",
    )
    recipe = priors.CausalPipelineRecipe(
        task=priors.LinearRecipe(
            rows=16,
            context_rows=8,
            features=1,
            signal_features=1,
            active_features=1,
            task_type=task_type,
            classes={"regression": 0, "binary": 2, "multiclass": 3}[task_type],
        ),
        nodes=(
            priors.RootInputCausalNode(),
            priors.ActivatedCausalNode(
                parents=(0,),
                weights=((1.0, -2.0, 0.5),),
                bias=(0.1, 0.2, 0.3),
                noise_scale=(0.1, 0.1, 0.1),
                noise_position="after_output",
            ),
        ),
        pipelines=(
            priors.CausalNodePipeline(channels=1),
            priors.CausalNodePipeline(channels=3, stages=(fitted,)),
        ),
        observed=((0, 0),),
        target=tuple((1, j) for j in range(3 if task_type == "multiclass" else 1)),
        normalize=True,
        calibration_rows=32,
        version="causal-v2",
    )
    original = ExplorerSession(tmp_path / "original")
    created = original.request(
        "generate", {"record": {"recipe": asdict(recipe), "seed": 28, "task_id": 29}}
    )
    graph = original.request("graph", {"indices": [0, 15]})["graph"]
    assert graph["target_max_error"] < 1e-12
    stage = next(node for node in graph["nodes"] if node["id"] == "pipeline-1-stage-0")
    assert (
        stage["parameters"]["pipeline"]["stages"][0]["calibration_id"]
        == "saved-independent-standardize"
    )
    opened = ExplorerSession(tmp_path / "original").request(
        "open", {"archive": created["archive"], "task_id": 29}
    )
    assert opened["view"] == created["view"]
    replay = ExplorerSession(tmp_path / "replay")
    repeated = replay.request("generate", {"record_json": created["recipe_json"]})
    assert repeated["recipe_json"] == created["recipe_json"]
    assert replay.request("row", {"index": 15}) == original.request("row", {"index": 15})
