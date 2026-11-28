"""Calibration is explicit across guarded curves, frozen identity and authored edits."""

import json
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import pytest

from r0.modules import priors as p
from r0.modules.explorer import ExplorerSession


def opened(path: Path):
    task = p.LinearRecipe(rows=16, context_rows=8, features=2, signal_features=2, active_features=2)
    fn = p.FrozenNetworkFunction(
        2, (p.NetworkBlock((0,), (p.NetworkAffine(((2.0, 3.0),), (0.1,)),)),)
    )
    base = p.NetworkRecipe(task, fn)
    match = p.fit_network_variance_match(
        base, before_inputs=(0, 1), after_inputs=(1,), rows=37, rng=np.random.default_rng(42)
    )
    view = p.MatchedNetworkInputView(
        base, replace(task, features=4, active_features=1), (1,), matching=match
    )
    session = ExplorerSession(path)
    result = session.request("generate", {"recipe": asdict(view), "seed": 9, "task_id": 2})
    return session, result, match


def test_matched_workbench_identity_metadata_and_guard(tmp_path: Path):
    session, before, match = opened(tmp_path)
    details = before["view"]["construction_details"]
    assert details["input_view"]["matching"]["rows"] == 37
    after = session.request(
        "generate",
        {"record_json": before["recipe_json"], "frozen_identity": {"seed": "9", "task_id": "3"}},
    )
    assert json.loads(after["recipe_json"])["recipe"] == json.loads(before["recipe_json"])["recipe"]
    with pytest.raises(ValueError, match="limit"):
        session.request("curve", {"feature": 0, "causal_limits": {"working_bytes": 1}})
    curve = session.request("curve", {"feature": 1, "points": 5, "bounds": [-1, 1]})["curve"]
    raw = 3 * np.array(curve["values"]) + 0.1
    expected = (raw - match.after_mean[0]) * match.before_std[0] / match.after_std[
        0
    ] + match.before_mean[0]
    np.testing.assert_allclose(np.array(curve["scores"])[:, 0], expected, atol=1e-13)


@pytest.mark.parametrize("policy", ["recalibrate", "raw"])
def test_matched_authored_edit_policy_is_explicit_and_preserves_population(
    tmp_path: Path, policy: str
):
    session, before, match = opened(tmp_path)
    editor = session.request("mechanism", {})
    assert editor["matching_edit"]
    draft = json.loads(json.dumps(editor["editable_recipe"]))
    assert draft["version"] == "network-input-view-v1"
    draft["base"]["network"]["blocks"][0]["experts"][0]["matrix"] = [[4.0, 5.0]]
    result = session.request("network_draft", {"recipe": draft, "matching_policy": policy})[
        "recipe"
    ]
    edited = p.recipe_from_dict(result)
    assert isinstance(edited, p.NetworkInputView)
    if policy == "recalibrate":
        assert isinstance(edited, p.MatchedNetworkInputView)
        assert edited.matching is not None
        assert edited.matching.population == match.population
        assert edited.matching.before_scores != match.before_scores
        assert edited.matching.initial == match.initial
    else:
        assert type(edited) is p.NetworkInputView
    saved = session.request("edit", {"recipe": result})
    assert saved["recipe_json"] == before["recipe_json"]
    assert saved["edit"]["after"]["recipe"]["recipe"]["version"] == result["version"]


def test_matched_invalid_policy_and_calibration_failure_preserve_baseline(tmp_path: Path):
    session, before, _ = opened(tmp_path)
    draft = session.request("mechanism", {})["editable_recipe"]
    with pytest.raises(ValueError, match="policy"):
        session.request("network_draft", {"recipe": draft, "matching_policy": "guess"})
    draft = json.loads(json.dumps(draft))
    draft["base"]["network"]["blocks"][0]["experts"][0]["matrix"] = [[0.0, 0.0]]
    with pytest.raises(ValueError, match="constant independent score"):
        session.request("network_draft", {"recipe": draft, "matching_policy": "recalibrate"})
    assert session.request("view", {})["recipe_json"] == before["recipe_json"]
