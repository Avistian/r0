"""Browser-number transport supports explicit authored edits, not archive coercion."""

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any, cast

import pytest

from r0.modules import generation as g
from r0.modules import priors as p
from r0.modules.explorer import ExplorerSession


def recipe() -> p.FunctionalCausalRecipe:
    tree = p.XGBoostVectorTree(
        (0, -1, -1),
        (0.0, 0.0, 0.0),
        (1, -1, -1),
        (2, -1, -1),
        (True, False, False),
        ((0.0,), (1.0,), (2.0,)),
    )
    function = p.FrozenXGBoostVector(1, (0.0,), (tree,))
    return p.FunctionalCausalRecipe(
        p.LinearRecipe(rows=16, context_rows=8, features=1, signal_features=1, active_features=1),
        (
            p.FunctionalCausalNode(
                (), p.FrozenLinearFunction(((1.0,),)), p.JointRootBase("gaussian", 1)
            ),
            p.FunctionalCausalNode((0,), function),
        ),
        ((0, 0),),
        ((1, 0),),
    )


def browser_numbers(value: Any) -> Any:
    if type(value) is float and value.is_integer():
        return int(value)
    if isinstance(value, dict):
        return {key: browser_numbers(item) for key, item in cast(dict[str, Any], value).items()}
    if isinstance(value, (list, tuple)):
        return [browser_numbers(item) for item in cast(list[Any] | tuple[Any, ...], value)]
    return value


def test_browser_noise_and_integer_leaf_edit_save_reopen(tmp_path: Path):
    session = ExplorerSession(tmp_path)
    original = session.request(
        "generate", {"recipe": asdict(recipe()), "seed": "19", "task_id": "18014398509481987"}
    )
    draft = browser_numbers(session.request("mechanism", {})["editable_recipe"])
    draft["nodes"][1]["noise_scale"] = [0.1]
    draft["nodes"][1]["function"]["trees"][0]["thresholds"][0] = 1
    draft["nodes"][1]["function"]["trees"][0]["values"][2][0] = 7
    # Exactly the same browser draft is still not an accepted stored archive.
    with pytest.raises(ValueError, match="vector|XGBoost"):
        p.recipe_from_dict(draft)
    edited = session.request("edit", {"recipe": draft})
    assert session.edit is not None
    after = json.loads(session.edit.after.recipe_json)["recipe"]
    assert edited["edit"]["card"]["after_episode_id"] != original["view"]["episode_id"]
    assert session.request("edit_export", {})["data"]
    reopened = session.request("open_edit", {"path": edited["edit"]["path"]})
    assert reopened["edit"]["card"] == edited["edit"]["card"]
    assert after["nodes"][1]["function"]["trees"][0]["values"][2][0] == 7.0
    assert type(after["nodes"][1]["function"]["trees"][0]["values"][2][0]) is float
    assert after["nodes"][1]["function"]["trees"][0]["thresholds"][0] == 1.0
    assert session.task is not None
    assert session.task.recipe_json == original["recipe_json"]


@pytest.mark.parametrize("field", ["initial", "thresholds", "values"])
def test_authored_xgboost_rejects_boolean_numbers(tmp_path: Path, field: str):
    draft = browser_numbers(asdict(recipe()))
    function = draft["nodes"][1]["function"]
    if field == "initial":
        function[field][0] = False
    elif field == "thresholds":
        function["trees"][0][field][0] = False
    else:
        function["trees"][0][field][0][0] = False
    job = g.run_guarded_document(
        {
            "recipe": draft,
            "edit_policy": "authored-mechanism-v1",
            "seed": 1,
            "start": 2,
            "count": 1,
            "group_size": 1,
            "max_seconds": 30,
            "max_bytes": 1024**2,
        },
        tmp_path,
    )
    assert job.returncode != 0


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -float("inf")])
def test_authored_xgboost_rejects_nonfinite(tmp_path: Path, bad: float):
    draft = browser_numbers(asdict(recipe()))
    draft["nodes"][1]["function"]["initial"][0] = bad
    with pytest.raises(ValueError, match="JSON compliant"):
        g.run_guarded_document(
            {
                "recipe": draft,
                "edit_policy": "authored-mechanism-v1",
                "max_seconds": 30,
            },
            tmp_path,
        )


def test_authored_transport_keeps_integer_indices_strict(tmp_path: Path):
    draft = browser_numbers(asdict(recipe()))
    draft["nodes"][1]["function"]["trees"][0]["features"][0] = 0.0
    job = g.run_guarded_document(
        {
            "recipe": draft,
            "edit_policy": "authored-mechanism-v1",
            "seed": 1,
            "start": 2,
            "count": 1,
            "group_size": 1,
            "max_seconds": 30,
            "max_bytes": 1024**2,
        },
        tmp_path,
    )
    assert job.returncode != 0
    assert "invalid vector tree field" in job.stderr
