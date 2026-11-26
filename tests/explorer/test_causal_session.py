"""Sampled causal tasks travel through the shared saved explorer workflow."""

import json
from pathlib import Path

import pytest

from r0.modules.explorer import ExplorerSession


@pytest.mark.parametrize("family", ["forward", "layered", "redirection"])
@pytest.mark.parametrize(
    ("profile", "construction"),
    [
        ("causal-affine-v1", "gaussian-affine-v1"),
        ("causal-activated-v1", "gaussian-activated-v1"),
        ("causal-preferences-v1", "gaussian-activation-prior-v1"),
        ("causal-exogenous-v1", "activation-exogenous-v1"),
    ],
)
def test_sampled_causal_session_saves_reopens_and_replays_its_graph(
    tmp_path: Path, family: str, profile: str, construction: str
) -> None:
    session = ExplorerSession(tmp_path / "original")
    result = session.request(
        "generate",
        {
            "record": {
                "recipe": {"rows": 16, "context_rows": 8},
                "seed": 28,
                "task_id": 10,
            },
            "sampling": {"family": family, "profile": profile},
        },
    )
    record = json.loads(result["recipe_json"])
    assert record["recipe"]["topology"]["prior"]["family"] == family
    assert record["recipe"]["construction"] == construction
    trace = session.request("graph", {"indices": [0, 15]})["graph"]
    assert trace["target_max_error"] < 1e-10
    assert any(node["id"] == "causal-0" for node in trace["nodes"])
    reopened = ExplorerSession(tmp_path / "original").request(
        "open", {"archive": result["archive"], "task_id": 10}
    )
    assert reopened["view"] == result["view"]
    replay_session = ExplorerSession(tmp_path / "replay")
    replay = replay_session.request("generate", {"record_json": result["recipe_json"]})
    assert replay["recipe_json"] == result["recipe_json"]
    assert replay_session.request("row", {"index": 15}) == session.request("row", {"index": 15})
