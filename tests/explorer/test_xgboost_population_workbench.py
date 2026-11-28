"""Actual-parent fits remain fixed through guarded replay and query continuation."""

import json
from pathlib import Path

import numpy as np
import pytest

from r0.modules.explorer import ExplorerSession


@pytest.mark.parametrize("family", ["functional", "native", "vector", "converted"])
@pytest.mark.parametrize(
    "population", ["support-only", "batch-refit-reference", "independent-ancestors"]
)
def test_population_xgboost_session_custody(tmp_path: Path, family: str, population: str):
    session = ExplorerSession(tmp_path)
    catalog = session.request("construction_examples", {})["examples"]
    controls = next(
        v
        for k, v in catalog.items()
        if k.startswith(f"Population XGBoost: {population} · {family} linear")
    )
    independent = population == "independent-ancestors"
    identity = {"seed": "79" if independent else "71", "task_id": "83" if independent else "73"}
    view = session.request(
        "construct", {"controls": controls, "construction_seed": "7", **identity}
    )
    assert session.task is not None
    before = session.task
    source = json.loads(before.recipe_json)["recipe"]
    info = session.request("xgboost_population_controls", {})
    assert info["xgboost_population"] is True
    replay = session.request("xgboost_population_replay", {})
    assert replay["recipe_json"] == view["recipe_json"]
    graph = session.request("graph", {"indices": [7, 2, 7]})["graph"]
    assert graph["rows"] == [7, 2, 7]
    if independent:
        query = session.request("xgboost_population_query", {"seed": "89", "task_id": "97"})
        assert query["view"]["recipe"]["recipe"] == view["view"]["recipe"]["recipe"]
        assert session.task is not None
        assert not np.array_equal(session.task.inputs, before.inputs)
    else:
        with pytest.raises(ValueError, match="explicit new fit"):
            session.request("xgboost_population_query", {"seed": "89", "task_id": "97"})
    current = session.request("view", {})
    with pytest.raises(ValueError, match="fit|Generation failed"):
        session.request("xgboost_population_query", {"seed": "71", "task_id": "73"})
    assert session.request("view", {})["recipe_json"] == current["recipe_json"]
    assert json.loads(current["recipe_json"])["recipe"]["fits"] == source["fits"]
