"""Read-only network admission rejects malformed drafts without replacing saved state."""

import json
from copy import deepcopy
from pathlib import Path
from typing import Any

import pytest

from r0.modules.explorer import ExplorerSession


def test_network_draft_validation_preserves_state_and_enforces_guards(tmp_path: Path):
    session = ExplorerSession(tmp_path)
    controls = session.request("construction_examples", {})["examples"][
        "Standalone network: sparse + gated + skip (authored)"
    ]
    controls["task"].update(rows=16, context_rows=8)
    original = session.request(
        "construct", {"controls": controls, "construction_seed": 1, "seed": 3, "task_id": 4}
    )
    draft = json.loads(json.dumps(session.request("mechanism", {})["editable_recipe"]))
    before = session.task
    files = sorted(str(p) for p in tmp_path.rglob("*"))
    invalid_blocks: tuple[Any, ...] = ({}, None, {**draft["network"]["blocks"][0], "experts": []})
    for block in invalid_blocks:
        bad = deepcopy(draft)
        bad["network"]["blocks"][0] = block
        with pytest.raises(ValueError, match="Invalid network draft"):
            session.request("network_draft", {"recipe": bad})
        assert session.task is before
        assert session.request("view", {})["recipe_json"] == original["recipe_json"]
    with pytest.raises(ValueError, match="timed out"):
        session.request("network_draft", {"recipe": draft, "max_seconds": 0.001})
    with pytest.raises(ValueError, match="byte cap"):
        session.request("network_draft", {"recipe": draft, "max_bytes": 1})
    with pytest.raises(ValueError, match="Invalid network draft"):
        session.request("network_draft", {"recipe": draft, "causal_limits": {"parameters": 1}})
    draft["network"]["blocks"][0]["experts"][0]["bias"][0] += 3
    validated = session.request("network_draft", {"recipe": draft})
    assert json.loads(json.dumps(validated["recipe"])) == draft
    assert session.task is before
    assert session.edit is None
    assert sorted(str(p) for p in tmp_path.rglob("*")) == files
