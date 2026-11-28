"""Guarded shared-callback delivery for stochastic standalone networks."""

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from r0.modules import priors as p
from r0.modules.explorer import ExplorerSession


def construct(session: ExplorerSession):
    examples = session.request("construction_examples", {})["examples"]
    return session.request(
        "construct",
        {
            "controls": examples["Noisy network: dense"],
            "construction_seed": "7",
            "seed": "11",
            "task_id": str(2**54 + 3),
        },
    )


def test_noisy_network_construct_replay_curve_and_authored_edit(tmp_path: Path):
    session = ExplorerSession(tmp_path)
    generated = construct(session)
    raw = json.loads(generated["recipe_json"])["recipe"]
    recipe = p.recipe_from_dict(raw)
    assert isinstance(recipe, p.NetworkInnovationRecipe)
    assert raw["sampling"] is not None
    editor = session.request("mechanism", {})
    assert editor["noise"]
    assert editor["editable_recipe"]["network"]["version"] == "frozen-structural-network-v1"
    before = session.request("view", {})
    zero = session.request("curve", {"feature": 0, "points": 5, "reference": "zero"})["curve"]
    fixed = session.request("curve", {"feature": 0, "points": 5, "reference": "original-row"})[
        "curve"
    ]
    assert zero["kind"] == "network_innovation_reference"
    assert fixed["original_task_id"] == str(2**54 + 3)
    x = torch.tensor(zero["anchor_inputs"], dtype=torch.float64).repeat(5, 1)
    x[:, 0] = torch.tensor(zero["values"], dtype=torch.float64)
    np.testing.assert_allclose(
        zero["scores"], p.evaluate_network_innovation_reference(recipe, x).numpy()
    )
    draws = p.network_innovation_row(recipe, seed=11, task_id=2**54 + 3, row=0)
    np.testing.assert_allclose(
        fixed["scores"],
        p.evaluate_network_innovation_reference(
            recipe, x, innovations={i: v.repeat(5, 1) for i, v in draws.items()}
        ).numpy(),
    )
    with pytest.raises(ValueError, match="Invalid network noise draft"):
        session.request(
            "network_noise_draft", {"recipe": editor["editable_recipe"], "innovation_plan": {}}
        )
    assert session.request("view", {}) == before
    editor["innovation_plan"]["importance"] = [1.0, 2.0]
    validated = session.request(
        "network_noise_draft",
        {"recipe": editor["editable_recipe"], "innovation_plan": editor["innovation_plan"]},
    )
    edited = session.request("edit", {"recipe": validated["compiled_recipe"]})
    assert edited["edit"]["card"]["same_seed_and_task_id"]
    next_result = session.request(
        "generate",
        {
            "record_json": generated["recipe_json"],
            "frozen_identity": {"seed": "11", "task_id": str(2**54 + 5)},
        },
    )
    assert json.loads(next_result["recipe_json"])["recipe"] == raw


def test_noise_worker_boundaries_and_timeout_preserve_baseline(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    session = ExplorerSession(tmp_path)
    construct(session)
    baseline = session.request("view", {})

    def forbidden(*args: object, **kwargs: object):
        raise AssertionError("expensive noisy recipe decoding ran in the parent")

    monkeypatch.setattr(p, "recipe_from_dict", forbidden)
    editor = session.request("mechanism", {})
    session.request("curve", {"feature": 0, "points": 3, "reference": "original-row"})
    session.request(
        "network_noise_draft",
        {"recipe": editor["editable_recipe"], "innovation_plan": editor["innovation_plan"]},
    )
    for action, data in (
        ("mechanism", {}),
        ("curve", {"feature": 0, "points": 3, "reference": "original-row"}),
        (
            "network_noise_draft",
            {"recipe": editor["editable_recipe"], "innovation_plan": editor["innovation_plan"]},
        ),
    ):
        with pytest.raises(ValueError, match="timed out"):
            session.request(action, {**data, "max_seconds": 0.001})
        assert session.request("view", {}) == baseline


@pytest.mark.parametrize("profile", ["p1-paper", "p1-778f0d0", "tabicl-9530bd6"])
def test_source_qualified_noise_presets(tmp_path: Path, profile: str):
    session = ExplorerSession(tmp_path)
    controls = session.request("construction_examples", {})["examples"][
        f"Source noise placement: {profile} (explicit importance adaptation)"
    ]
    result = session.request(
        "construct", {"controls": controls, "construction_seed": 7, "seed": 11, "task_id": 3}
    )
    raw = json.loads(result["recipe_json"])["recipe"]
    assert raw["plan"]["profile"] == profile + "-placement"
    assert (raw["plan"]["stages"][0] is None) == (profile != "p1-paper")
    assert "not claim full prior" in result["view"]["construction_details"]["notice"]


def test_compiled_noise_intervention_and_task_pair_delivery(tmp_path: Path):
    session = ExplorerSession(tmp_path)
    original = construct(session)
    raw = json.loads(original["recipe_json"])["recipe"]
    controls = session.request("intervention_controls", {})
    assert len(controls["nodes"]) == len(raw["nodes"])
    result = session.request(
        "intervene",
        {
            "interventions_json": json.dumps(
                [
                    {
                        "version": "hard-causal-intervention-v1",
                        "node": raw["target"][0][0],
                        "channels": [0],
                        "values": [0.7],
                        "category": None,
                        "location": "state_and_readout",
                    }
                ]
            ),
            "mode": "matched",
            "population": 1,
        },
    )
    assert result["intervention"] is not None
    assert session.request("view", {})["recipe_json"] == original["recipe_json"]
    pair = session.request("pair", {"intervention": "noise", "value": 0.5})
    assert pair["pair"] is not None
