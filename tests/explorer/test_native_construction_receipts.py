"""Malformed construction geometry must retain failure receipts and the active task."""

import json
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path
from typing import Any

import pytest

from r0.modules.explorer import ExplorerSession
from r0.modules.priors import LinearRecipe


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("rows", 15),
        ("context_rows", 16),
        ("observed", None),
        ("observed", []),
        ("observed", [[999, 0]]),
    ],
)
def test_malformed_native_geometry_has_failed_receipt_and_preserves_task(
    tmp_path: Path, field: str, value: Any
) -> None:
    session = ExplorerSession(tmp_path)
    session.request(
        "generate",
        {
            "recipe": asdict(
                LinearRecipe(
                    rows=16, context_rows=8, features=1, signal_features=1, active_features=1
                )
            ),
            "seed": 2,
            "task_id": 3,
        },
    )
    before = session.request("view", {})
    controls = session.request("construction_examples", {})["examples"][
        "Native mixed graph (sampled domains)"
    ]
    controls.update(rows=16, context_rows=8, calibration_rows=16)
    controls[field] = deepcopy(value)
    receipts_before = set(tmp_path.glob(".attempts/**/*-result.json"))
    with pytest.raises(ValueError, match="Generation failed"):
        session.request(
            "construct",
            {
                "controls": controls,
                "construction_seed": 28,
                "seed": 19,
                "task_id": 7,
            },
        )
    added = set(tmp_path.glob(".attempts/**/*-result.json")) - receipts_before
    assert len(added) == 1
    receipt = json.loads(added.pop().read_text())
    assert receipt["status"] == "FAILED"
    assert receipt["result"]["returncode"] != 0
    assert json.loads(receipt["request"]["construction"]["controls_json"])[field] == value
    assert session.request("view", {}) == before


def test_construct_does_not_decode_task_geometry_in_parent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from r0.modules.generation import GraphConstruction

    def parent_decode_forbidden(self: GraphConstruction) -> dict[str, Any]:
        raise AssertionError("construction task decoding must run in the guarded subprocess")

    monkeypatch.setattr(GraphConstruction, "task_record", parent_decode_forbidden)
    session = ExplorerSession(tmp_path)
    controls = session.request("construction_examples", {})["examples"][
        "Native mixed graph (sampled domains)"
    ]
    controls.update(rows=16, context_rows=8, calibration_rows=16)
    result = session.request(
        "construct",
        {
            "controls": controls,
            "construction_seed": 28,
            "seed": 19,
            "task_id": 7,
        },
    )
    assert result["view"]["target_origin"] == "native-variable-v1"
