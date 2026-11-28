"""Guarded fitting validates every selection before starting a backend."""

import json
from dataclasses import asdict
from typing import Any

import pytest

from r0.modules import priors as p
from r0.modules.generation import GraphConstruction


def controls() -> dict[str, Any]:
    skeleton = p.FunctionalCausalRecipe(
        p.LinearRecipe(rows=16, context_rows=8, features=1, signal_features=1, active_features=1),
        (
            p.FunctionalCausalNode(
                (), p.FrozenLinearFunction(((1.0,),)), p.JointRootBase("gaussian", 1)
            ),
            p.FunctionalCausalNode((0,), p.FrozenLinearFunction(((1.0,), (2.0,)))),
        ),
        ((0, 0),),
        ((1, 0),),
    )
    plan = p.XGBoostFitPlan(input_width=1, output_width=2, rows=32, estimators=1, depth=1)
    return {
        "topology": "xgboost-fitted",
        "skeleton": asdict(skeleton),
        "fits": [{"node": 1, "plan": asdict(plan)}],
    }


def test_guarded_xgboost_replays_complete_fit():
    raw = controls()
    request = GraphConstruction(json.dumps(raw), 91)
    assert request.task_record() == json.loads(json.dumps(raw["skeleton"]["task"]))
    first = request.build(p.CausalLimits())
    second = request.build(p.CausalLimits())
    assert isinstance(first, p.XGBoostFittedGraph)
    assert first == second
    assert p.recipe_from_dict(asdict(first)) == first


@pytest.mark.parametrize("bad", ["duplicate", "geometry", "index", "boolean"])
def test_all_fit_selections_checked_before_backend(bad: str, monkeypatch: pytest.MonkeyPatch):
    raw = controls()
    entry = json.loads(json.dumps(raw["fits"][0]))
    if bad == "geometry":
        entry["plan"]["input_width"] = 2
    elif bad == "index":
        entry["node"] = 999
    elif bad == "boolean":
        entry["node"] = True
    if bad != "duplicate":
        raw["fits"][0]["node"] = 0
        raw["fits"][0]["plan"]["output_width"] = 1
    raw["fits"].append(entry)
    calls: list[bool] = []

    def forbidden(*args: object, **kwargs: object):
        calls.append(True)
        raise AssertionError("backend started before all selections were validated")

    monkeypatch.setattr(p, "fit_xgboost", forbidden)
    with pytest.raises(ValueError, match="XGBoost"):
        GraphConstruction(json.dumps(raw), 91).build(p.CausalLimits())
    assert calls == []
