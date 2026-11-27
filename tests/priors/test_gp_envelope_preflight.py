"""Invalid outer GP envelopes must not execute their retained population."""

import copy
from dataclasses import asdict
from typing import Any

import numpy as np
import pytest

from r0.modules import priors as p


@pytest.fixture
def packets() -> tuple[dict[str, Any], dict[str, Any]]:
    pop = p.realize_gp_columns(p.GPColumnRequest(rows=16), seed=17, task_id=23)
    task = p.LinearRecipe(
        rows=16, context_rows=10, features=2, signal_features=2, active_features=2
    )
    target = p.GPColumnTarget("linear", ((1.0,), (2.0,)), (0.0,))
    return asdict(p.GPColumnRecipe(pop, task)), asdict(p.compare_gp_columns(pop, target))


@pytest.mark.parametrize(
    "case",
    [
        "recipe-version",
        "task-rows",
        "task-features",
        "task-active",
        "task-outputs",
        "task-version",
        "source-version",
        "population-version",
        "population-seed",
        "pair-version",
        "pair-mode",
        "pair-fresh-missing",
        "pair-fresh-bool",
        "pair-fresh-negative",
        "pair-matched-seed",
        "replacement-width",
        "replacement-outputs",
        "replacement-version",
        "missing-field",
        "extra-field",
    ],
)
def test_bad_outer_controls_reject_before_rng(
    case: str, packets: tuple[dict[str, Any], dict[str, Any]], monkeypatch: pytest.MonkeyPatch
) -> None:
    recipe, pair = copy.deepcopy(packets)
    target = recipe
    decode = p.gp_column_recipe_from_dict
    if case.startswith(("pair-", "replacement-", "population-")):
        target, decode = pair, p.gp_column_comparison_from_dict
    if case == "recipe-version":
        recipe["version"] = "invalid"
    elif case == "task-rows":
        recipe["task"]["rows"] = 17
    elif case == "task-features":
        recipe["task"]["features"] = 3
    elif case == "task-active":
        recipe["task"]["active_features"] = 1
    elif case == "task-outputs":
        recipe["task"].update(task_type="multiclass", classes=3)
    elif case == "task-version":
        recipe["task"]["version"] = "invalid"
    elif case == "source-version":
        recipe["source"]["version"] = "invalid"
    elif case == "population-version":
        pair["original"]["version"] = "invalid"
    elif case == "population-seed":
        pair["original"]["seed"] = True
    elif case == "pair-version":
        pair["version"] = "invalid"
    elif case == "pair-mode":
        pair["mode"] = "invalid"
    elif case == "pair-fresh-missing":
        pair["mode"] = "fresh"
    elif case == "pair-fresh-bool":
        pair.update(mode="fresh", fresh_seed=True)
    elif case == "pair-fresh-negative":
        pair.update(mode="fresh", fresh_seed=-1)
    elif case == "pair-matched-seed":
        pair["fresh_seed"] = 1
    elif case == "replacement-width":
        pair["replacement"]["weights"] = ((1.0,),)
    elif case == "replacement-outputs":
        pair["replacement"].update(weights=((1.0, 2.0), (3.0, 4.0)), bias=(0.0, 0.0))
    elif case == "replacement-version":
        pair["replacement"]["version"] = "invalid"
    elif case == "missing-field":
        del recipe["task"]
    elif case == "extra-field":
        recipe["unexpected"] = True
    calls: list[int] = []
    original = np.random.default_rng

    def tracked(*args: Any, **kwargs: Any) -> Any:
        calls.append(1)
        return original(*args, **kwargs)

    monkeypatch.setattr(np.random, "default_rng", tracked)
    with pytest.raises(ValueError, match="GP"):
        decode(target)
    assert calls == []
