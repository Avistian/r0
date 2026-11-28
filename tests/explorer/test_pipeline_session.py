"""Explicit state/readout pipeline variants survive saved explorer workflows."""

from dataclasses import asdict
from pathlib import Path

import numpy as np
import pytest

from r0.modules import priors
from r0.modules.episodes import TaskType
from r0.modules.explorer import ExplorerSession


@pytest.mark.parametrize("variant", ["historical-tuple", "historical-graph", "readout-only"])
@pytest.mark.parametrize("task_type", ["regression", "binary", "multiclass"])
@pytest.mark.parametrize("root_input", [False, True])
def test_pipeline_saves_reopens_and_traces(
    tmp_path: Path, variant: str, task_type: TaskType, root_input: bool
) -> None:
    weights = ((1.0, 0.5, -1.0), (-0.25, 1.0, 2.0))
    construction = priors.FactorizedWeights(
        left=((1.0, 0.0), (0.0, 1.0)), right=weights, diagonal=(1.0, 1.0)
    )
    warp = priors.calibrate_kumaraswamy(
        np.array([[-2.0], [2.0]]), a=0.7, b=1.3, calibration_id="independent-converter-range"
    )
    activation = priors.calibrate_source_activation(
        priors.CausalActivation("square"),
        np.array([[-2.0], [-1.0], [1.0], [2.0]]),
        rng=np.random.default_rng(131),
        scale=1.0,
        calibration_id="saved-activation-converter",
    )
    recipe = priors.CausalPipelineRecipe(
        task=priors.LinearRecipe(
            rows=16,
            context_rows=8,
            features=2,
            signal_features=2,
            active_features=2,
            task_type=task_type,
            classes={"regression": 0, "binary": 2, "multiclass": 3}[task_type],
        ),
        nodes=(
            priors.RootInputCausalNode(
                bias=(0.25, 0.75),
                root_scale=(1.0, -2.0),
                noise_scale=(0.3, 0.5),
                root_exogenous=priors.ExogenousDistribution(
                    inputs=priors.InputDistribution(distribution="mixture")
                ),
            )
            if root_input
            else priors.ActivatedCausalNode(
                bias=(0.25, 0.75), noise_scale=(1.0, 1.0), noise_position="after_output"
            ),
            priors.MatrixCausalNode(
                parents=(0,),
                weights=weights,
                bias=(0.0, 0.0, 0.0),
                noise_scale=(0.2, 0.2, 0.2),
                weight_construction=construction,
                noise_position="after_output",
            ),
        ),
        pipelines=(
            priors.CausalNodePipeline(
                channels=2,
                converters=(priors.NumericalConverter(channel=0, transform=warp, variant=variant),)
                + (
                    (priors.NumericalConverter(channel=1, transform=activation, variant=variant),)
                    if root_input
                    else ()
                ),
            ),
            priors.CausalNodePipeline(
                channels=3,
                converters=(priors.NumericalConverter(channel=0, transform=warp, variant=variant),),
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
    graph = session.request("graph", {"indices": [0, 15]})["graph"]
    assert graph["target_max_error"] < 1e-12
    assert next(n for n in graph["nodes"] if n["id"] == "inputs")["parents"] == ["readout-0"]
    reopened = ExplorerSession(tmp_path / "original").request(
        "open", {"archive": result["archive"], "task_id": 0}
    )
    assert reopened["view"] == result["view"]
    replay = ExplorerSession(tmp_path / "replay")
    repeated = replay.request("generate", {"record_json": result["recipe_json"]})
    assert repeated["recipe_json"] == result["recipe_json"]
    assert replay.request("row", {"index": 15}) == session.request("row", {"index": 15})
