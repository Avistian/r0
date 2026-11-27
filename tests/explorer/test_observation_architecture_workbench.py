"""General architecture observation controls retain explicit extension provenance."""

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import pytest
import torch

from r0.modules import generation as g
from r0.modules import priors as p
from r0.modules.explorer import ExplorerSession


def choices(session: ExplorerSession) -> dict[str, Any]:
    return {
        k: v
        for k, v in session.request("construction_examples", {})["examples"].items()
        if k.startswith("Observations: r0 general /")
    }


def test_general_observation_presets_replay_and_reference(tmp_path: Path) -> None:
    catalog = choices(ExplorerSession(tmp_path))
    assert len(catalog) == 12
    for name, controls in catalog.items():
        recipe = g.GraphConstruction(json.dumps(controls), 7).build(p.CausalLimits())
        assert isinstance(recipe, p.NetworkObservationRecipe), name
        assert recipe.observations.request.profile == "r0-propagated-block-v1"
        asdict_recipe = asdict(recipe)
        assert p.recipe_from_dict(asdict_recipe) == recipe
        task = p.generate_task(recipe, seed=37, task_id=2**54 + 17)
        assert p.explain_generation(task, [31, 0, 31])["rows"] == [31, 0, 31]
        ref = p.evaluate_observation_reference(
            recipe, task.inputs[4:5], seed=37, task_id=2**54 + 17, row=4
        )
        torch.testing.assert_close(ref, task.scores[4:5], atol=1e-12, rtol=1e-12)
        assert asdict_recipe["observations"]["request"]["profile"] == "r0-propagated-block-v1"


def test_guarded_general_observation_workflow_and_paper_rejection(tmp_path: Path) -> None:
    session = ExplorerSession(tmp_path)
    controls = choices(session)[
        "Observations: r0 general / authored sparse gated multi-parent skips"
    ]
    baseline = session.request(
        "construct",
        {"controls": controls, "construction_seed": "7", "seed": "37", "task_id": str(2**54 + 17)},
    )
    assert baseline["view"]["recipe"]["task_id"] == str(2**54 + 17)
    editor = session.request("mechanism", {})
    assert editor["observation_selection"]["profile"] == "r0-propagated-block-v1"
    assert len(editor["editable_recipe"]["nodes"][2]["function"]["block"]["experts"]) == 2
    curve = session.request("curve", {"feature": 0, "points": 3})["curve"]
    assert curve["recipe_json"] == baseline["recipe_json"]
    session.request("graph", {"indices": [31, 0, 31]})
    controls["observations"]["profile"] = "p1-778f0d0-v1"
    with pytest.raises(ValueError, match="Generation failed"):
        session.request(
            "construct",
            {
                "controls": controls,
                "construction_seed": "7",
                "seed": "37",
                "task_id": str(2**54 + 17),
            },
        )
    assert session.request("view", {}) == baseline
    with pytest.raises(ValueError, match="timed out"):
        session.request("mechanism", {"max_seconds": 0.001})
    assert session.request("view", {}) == baseline
    reopened = session.request("open", {"archive": baseline["archive"], "task_id": str(2**54 + 17)})
    assert reopened["recipe_json"] == baseline["recipe_json"]
