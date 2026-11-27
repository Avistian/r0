"""Bounded causal workflow preserves active datasets and completed comparisons on failure."""

import base64
import io
import json
import zipfile
from dataclasses import asdict
from pathlib import Path
from typing import Any

import pytest

from r0.modules import priors as p
from r0.modules.explorer import ExplorerSession
from r0.modules.generation import load_causal_pair


def original(session: ExplorerSession) -> dict[str, Any]:
    base = p.FunctionalCausalRecipe(
        p.LinearRecipe(rows=16, context_rows=8, features=2, signal_features=2, active_features=2),
        (
            p.FunctionalCausalNode(
                (), p.FrozenLinearFunction(((1.0,),)), p.JointRootBase("gaussian", 1)
            ),
            p.FunctionalCausalNode((0,), p.FrozenLinearFunction(((2.0,),)), noise_scale=(0.2,)),
            p.FunctionalCausalNode((0, 1), p.FrozenLinearFunction(((-1.0, 3.0),))),
        ),
        ((0, 0), (1, 0)),
        ((2, 0),),
    )
    return session.request(
        "generate",
        {
            "recipe": asdict(base),
            "seed": "9007199254740997",
            "task_id": "9007199254740999",
            "max_seconds": 30,
            "max_bytes": 2_000_000,
        },
    )


def test_sampled_policy_guarded_receipt_archive_and_failure_preservation(tmp_path: Path) -> None:
    session = ExplorerSession(tmp_path)
    baseline = original(session)
    draft = {"streams": [3, 5, 7], "reference_rows": 16}
    created = session.request(
        "sample_intervention_policy",
        {
            "policy_json": json.dumps(draft),
            "max_seconds": 30,
        },
    )
    pair = created["intervention"]
    assert created["recipe_json"] == baseline["recipe_json"]
    loaded = load_causal_pair(tmp_path / pair["path"])
    assert (
        json.loads(loaded.after.recipe_json)["recipe"]["version"]
        == "sampled-intervention-policy-v1"
    )
    assert session.task is not None
    assert loaded.before.recipe_json == session.task.recipe_json
    graph = session.request(
        "intervention_graph",
        {
            "indices": [15, 0, 15],
            "expected_intervention_id": pair["after_id"],
            "max_seconds": 30,
        },
    )
    assert graph["after"]["rows"] == [15, 0, 15]
    assert any(n["id"] == "policy-selection" for n in graph["after"]["nodes"])
    for override in (
        {"policy_json": "invalid"},
        {"max_seconds": 0.000001},
        {"causal_limits": {"working_bytes": 1}},
    ):
        with pytest.raises(ValueError, match="failed|limit"):
            session.request(
                "sample_intervention_policy",
                {
                    "policy_json": json.dumps(draft),
                    "max_seconds": 30,
                    **override,
                },
            )
        assert session.task is not None
        assert session.task.recipe_json == loaded.before.recipe_json
        assert session.intervention is not None
        assert session.intervention.after.episode_id == pair["after_id"]
    reopened = ExplorerSession(tmp_path).request(
        "open_intervention", {"path": pair["path"], "max_seconds": 30}
    )
    assert reopened["intervention"] == pair
    exported = session.request(
        "intervention_export", {"expected_intervention_id": pair["after_id"], "max_seconds": 30}
    )
    with zipfile.ZipFile(io.BytesIO(base64.b64decode(exported["data"]))) as archive:
        assert archive.read("after.zip") == (tmp_path / pair["path"] / "after.zip").read_bytes()


@pytest.mark.parametrize("family", ["normal-r0", "uniform-r0", "log-uniform-r0", "choice-r0"])
def test_soft_policy_drafts_preserve_receipt_on_next(tmp_path: Path, family: str) -> None:
    session = ExplorerSession(tmp_path)
    original(session)
    draft = {
        "streams": [3, 5, 7],
        "kind": "soft",
        "profile": "explicit-policy-r0-v1",
        "eligibility": [1],
        "count": 1,
        "mode": "matched",
        "parameter_laws": [
            {
                "node": 1,
                "path": ["matrix"],
                "family": family,
                "location": 0.1,
                "scale": 2.0,
                "choices": [-2.0, 4.0] if family == "choice-r0" else [],
            }
        ],
    }
    created = session.request(
        "sample_intervention_policy",
        {
            "policy_json": json.dumps(draft),
            "max_seconds": 30,
        },
    )
    pair = created["intervention"]
    opened = session.request(
        "open", {"archive": pair["path"] + "/after.zip", "task_id": "9007199254740999"}
    )
    advanced = session.request(
        "generate",
        {
            "record_json": opened["recipe_json"],
            "frozen_identity": {"seed": "9007199254740997", "task_id": "9007199254741001"},
            "max_seconds": 30,
        },
    )
    before = json.loads(opened["recipe_json"])
    after = json.loads(advanced["recipe_json"])
    assert before["recipe"] == after["recipe"]
    assert after["task_id"] == 9007199254741001
    details = advanced["view"]["construction_details"]
    assert details["bounds_lifecycle"] == "not-used-soft"
    assert details["parameter_laws"][0]["family"] == family


def test_source_reference_policy_uses_exact_original_identity(tmp_path: Path) -> None:
    session = ExplorerSession(tmp_path)
    original(session)
    created = session.request(
        "sample_intervention_policy",
        {
            "policy_json": json.dumps(
                {"streams": [3, 5, 7], "bounds_lifecycle": "source-reference-population"}
            ),
            "max_seconds": 30,
        },
    )
    recipe = created["intervention"]["after"]["recipe"]["recipe"]
    assert recipe["request"]["reference_seed"] == "9007199254740997"
    assert recipe["request"]["reference_task_id"] == "9007199254740999"
    assert recipe["request"]["reference_rows"] == 16
    assert recipe["request"]["mode"] == "population"
