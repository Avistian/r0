"""Authored edit decoding and replay belong to the killable generation worker."""

import json
from dataclasses import asdict, replace
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from r0.modules import priors as p
from r0.modules.explorer import ExplorerSession


def opened(path: Path):
    base = p.NetworkRecipe(
        p.LinearRecipe(rows=16, context_rows=7, features=2, signal_features=2, active_features=2),
        p.FrozenNetworkFunction(
            2, (p.NetworkBlock((0,), (p.NetworkAffine(((2.0, 3.0),), (0.1,)),)),)
        ),
    )
    matching = p.fit_network_variance_match(
        base, before_inputs=(0, 1), after_inputs=(1,), rows=37, rng=np.random.default_rng(42)
    )
    view = p.MatchedNetworkInputView(
        base, replace(base.task, active_features=1), (1,), matching=matching
    )
    session = ExplorerSession(path)
    original = session.request("generate", {"recipe": asdict(view), "seed": 19, "task_id": 7})
    draft = session.request("mechanism", {})["editable_recipe"]
    draft["base"]["network"]["blocks"][0]["experts"][0]["matrix"] = ((4.0, 5.0),)
    edited = session.request("network_draft", {"recipe": draft, "matching_policy": "recalibrate"})
    return session, original, edited["recipe"]


@pytest.mark.parametrize("timeout", [False, True])
def test_matched_edit_never_replays_calibration_in_parent(tmp_path: Path, timeout: bool):
    session, original, recipe = opened(tmp_path)

    def forbidden(*args: object, **kwargs: object) -> None:
        raise AssertionError("calibration replay reached parent outside guarded worker")

    with patch.dict(p.fit_network_variance_match.__globals__, draw_root_base=forbidden):
        if timeout:
            with pytest.raises(ValueError, match="Timed out: True"):
                session.request("edit", {"recipe": recipe, "max_seconds": 0.001})
            assert session.request("view", {})["recipe_json"] == original["recipe_json"]
        else:
            result = session.request("edit", {"recipe": recipe})
            assert result["recipe_json"] == original["recipe_json"]
            assert result["edit"]["after"]["recipe"]["recipe"] == json.loads(json.dumps(recipe))
            reopened = ExplorerSession(tmp_path).request(
                "open_edit", {"path": result["edit"]["path"]}
            )
            assert reopened["edit"]["card"] == result["edit"]["card"]
            assert reopened["recipe_json"] == original["recipe_json"]
    assert list((tmp_path / ".attempts").glob("*/*-result.json"))


def test_invalid_edit_is_guarded_and_preserves_previous_edit(tmp_path: Path):
    session, original, recipe = opened(tmp_path)
    prior = session.request("edit", {"recipe": recipe})
    bad = json.loads(json.dumps(recipe))
    bad["matching"]["after_mean"][0] += 1
    for document in (bad, asdict(p.LinearRecipe())):
        with pytest.raises(ValueError, match="Generation failed"):
            session.request("edit", {"recipe": document})
        result = session.request("view", {})
        assert result["recipe_json"] == original["recipe_json"]
        assert result["edit"]["card"] == prior["edit"]["card"]
    failures = [json.loads(p.read_text()) for p in (tmp_path / ".attempts").glob("*/*-result.json")]
    assert sum(r["status"] == "FAILED" for r in failures) == 2
