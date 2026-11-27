"""Functional mechanisms use the shared create/save/reopen/inspection session."""

import json
from dataclasses import asdict
from pathlib import Path

import pytest

from r0.modules import priors
from r0.modules.episodes import TaskType
from r0.modules.explorer import ExplorerSession


@pytest.mark.parametrize("calibrated", [False, True])
@pytest.mark.parametrize("task_type", ["regression", "binary", "multiclass"])
@pytest.mark.parametrize("family", ["uniform", "gaussian", "ball", "covariance"])
def test_functional_graph_saves_reopens_and_replays(
    tmp_path: Path,
    family: str,
    calibrated: bool,
    task_type: TaskType,
) -> None:
    base = (
        priors.JointRootBase(family, 2)
        if family != "covariance"
        else priors.JointRootBase(
            family,
            2,
            gaussian_factor=((1.0, 0.5), (0.0, 2.0)),
            weights=priors.UniformWeightVector((0.5, 0.5)),
            factor=((0.5, 0.25), (0.0, 1.0)),
            covariance_base="gaussian",
        )
    )
    kind = priors.CalibratedFunctionalCausalRecipe if calibrated else priors.FunctionalCausalRecipe
    recipe = kind(
        priors.LinearRecipe(
            rows=16,
            context_rows=8,
            features=2,
            signal_features=2,
            active_features=2,
            task_type=task_type,
            classes={"regression": 0, "binary": 2, "multiclass": 3}[task_type],
        ),
        (
            priors.FunctionalCausalNode(
                (), priors.FrozenLinearFunction(((1.0, 0.0), (0.0, 1.0))), root=base
            ),
            priors.FunctionalCausalNode(
                (0,),
                priors.FrozenQuadraticFunction(
                    2,
                    (0, 1),
                    (
                        ((1.0, 0.0, 0.0), (0.0, 0.0, 0.0), (0.0, 0.0, 0.0)),
                        ((0.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 0.0)),
                        ((0.0, 1.0, 0.0), (0.0, 0.0, 0.0), (0.0, 0.0, 0.0)),
                    ),
                ),
                noise_scale=(0.1, 0.2, 0.3),
            ),
        ),
        ((0, 0), (0, 1)),
        tuple((1, i) for i in range(3 if task_type == "multiclass" else 1)),
    )
    session = ExplorerSession(tmp_path / "original")
    result = session.request(
        "generate", {"record": {"recipe": asdict(recipe), "seed": 28, "task_id": 0}}
    )
    graph = session.request("graph", {"indices": [15, 1, 15]})["graph"]
    assert graph["signal_max_error"] == graph["target_max_error"] == 0
    assert json.loads(result["recipe_json"])["recipe"]["version"] == recipe.version
    reopened_session = ExplorerSession(tmp_path / "original")
    reopened = reopened_session.request("open", {"archive": result["archive"], "task_id": 0})
    assert reopened["view"] == result["view"]
    assert reopened_session.request("graph", {"indices": [15, 1, 15]})["graph"] == graph
    replay = ExplorerSession(tmp_path / "replay")
    repeated = replay.request("generate", {"record_json": result["recipe_json"]})
    assert repeated["recipe_json"] == result["recipe_json"]
    assert replay.request("row", {"index": 15}) == session.request("row", {"index": 15})
