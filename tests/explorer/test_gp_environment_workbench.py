"""Guarded references and linked edits retain selected comparison sides."""

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from r0.modules.explorer import ExplorerSession


def test_gp_environment_references_linked_after_and_failure_preservation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    session = ExplorerSession(tmp_path)
    controls = session.request("construction_examples", {})["examples"][
        "GP columns: neural / binary / per-column RBF"
    ]
    baseline = session.request(
        "construct",
        {
            "controls": controls,
            "construction_seed": 1,
            "seed": str(2**54 + 1),
            "task_id": str(2**54 + 3),
        },
    )
    target = json.loads(baseline["recipe_json"])["recipe"]["source"]["target"]
    target["weights"][0][0] += 0.7
    first = session.request(
        "gp_target_edit",
        {"target_json": json.dumps(target), "mode": "fresh", "fresh_seed": str(2**54 + 5)},
    )["intervention"]

    def forbidden(*args: Any, **kwargs: Any):
        raise AssertionError("GP environment replay escaped guard")

    monkeypatch.setattr(np.linalg, "cholesky", forbidden)
    for side in ("before", "after"):
        ref = session.request(
            "curve",
            {
                "feature": 0,
                "anchor_row": 5,
                "points": 5,
                "gp_reference_side": side,
                "expected_intervention_id": first["after_id"],
            },
        )["curve"]
        assert ref["environment_side"] == ("original" if side == "before" else "replacement")
        assert ref["anchor_inputs"] == first[side]["rows"][5]["inputs"]
        assert ref["fresh_seed"] == str(2**54 + 5)
    target["weights"][0][0] += 0.8
    second = session.request(
        "gp_target_edit",
        {
            "target_json": json.dumps(target),
            "mode": "retain",
            "gp_edit_parent": "after",
            "expected_intervention_id": first["after_id"],
        },
    )["intervention"]
    assert second["before_id"] == first["after_id"]
    assert second["after"]["recipe"]["recipe"]["version"] == "gp-column-target-link-v1"
    assert session.request("view", {})["recipe_json"] == baseline["recipe_json"]
    for data in (
        {"mode": "fresh"},
        {"fresh_seed": 1},
        {"target_json": "{}"},
        {"max_seconds": 0.001},
    ):
        with pytest.raises(ValueError, match="Intervention|GP|failed"):
            session.request(
                "gp_target_edit",
                {
                    "target_json": json.dumps(target),
                    "mode": "retain",
                    "gp_edit_parent": "after",
                    "expected_intervention_id": second["after_id"],
                    **data,
                },
            )
        assert session.request("view", {})["intervention"]["path"] == second["path"]
    ref = session.request(
        "curve",
        {
            "feature": 1,
            "points": 7,
            "gp_reference_side": "after",
            "expected_intervention_id": second["after_id"],
        },
    )["curve"]
    assert ref["environment_side"] == "linked-replacement"
    with pytest.raises(ValueError, match="timed out"):
        session.request(
            "curve",
            {
                "feature": 1,
                "gp_reference_side": "after",
                "expected_intervention_id": second["after_id"],
                "max_seconds": 0.001,
            },
        )
    session.request("open_intervention", {"path": second["path"]})
    assert (
        session.request("curve", {"feature": 0, "points": 3})["curve"]["environment_side"]
        == "replacement"
    )
    assert session.request("intervention_export", {"expected_intervention_id": second["after_id"]})[
        "data"
    ]
