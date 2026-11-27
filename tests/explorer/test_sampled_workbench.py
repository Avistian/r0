"""The workbench retains sampled construction, explicit caps and linked edits."""

import json
from pathlib import Path

import pytest

from r0.modules.explorer import ExplorerSession


@pytest.mark.parametrize(
    ("example", "version"),
    [
        ("Cauchy causal graph", "sampled-topology-graph-v1"),
        ("Geometry-sampled causal graph", "sampled-geometry-graph-v1"),
    ],
)
def test_sampled_workbench_constructs_edits_and_reopens(
    tmp_path: Path, example: str, version: str
) -> None:
    session = ExplorerSession(tmp_path)
    examples = session.request("construction_examples", {})["examples"]
    controls = examples[example]
    if "width_prior" in controls:
        controls["width_prior"].update(low=1, high=3)
    controls["task"].update(rows=16, context_rows=8)
    controls["calibration_rows"] = 4
    controls["function_prior"]["functions"] = "lin"
    caps = {"parameters": 1_000_001}
    result = session.request(
        "construct",
        {
            "controls": controls,
            "construction_seed": 28,
            "seed": 19,
            "task_id": 7,
            "causal_limits": caps,
            "max_seconds": 120,
        },
    )
    assert json.loads(result["recipe_json"])["recipe"]["version"] == version
    assert session.request("graph", {"indices": [7, 0, 7], "causal_limits": caps})["graph"][
        "rows"
    ] == [7, 0, 7]
    editor = session.request("mechanism", {"causal_limits": caps})
    assert editor["construction"] == version
    assert editor["kind"] == "mechanism-edit"
    edited = editor["editable_recipe"]
    edited["nodes"][0]["noise_scale"] = [0.2] * editor["nodes"][0]["width"]
    comparison = session.request(
        "edit", {"recipe": edited, "causal_limits": caps, "max_seconds": 120}
    )
    assert comparison["edit"]["card"]["kind"] == "mechanism-edit"
    assert comparison["archive"] == result["archive"]
    reopened = ExplorerSession(tmp_path).request(
        "open_edit", {"path": comparison["edit"]["path"], "causal_limits": caps}
    )
    assert reopened["edit"]["card"] == comparison["edit"]["card"]
    assert reopened["view"]["episode_id"] == result["view"]["episode_id"]


def test_graph_construction_runs_in_worker_and_timeout_keeps_active_task(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from r0.modules.generation import GraphConstruction

    session = ExplorerSession(tmp_path)
    controls = session.request("construction_examples", {})["examples"]["Explicit causal graph"]
    controls["task"].update(rows=16, context_rows=8)
    controls["calibration_rows"] = 4

    def parent_build(*args: object, **kwargs: object) -> object:
        raise AssertionError("construction must not run in the parent")

    monkeypatch.setattr(GraphConstruction, "build", parent_build)
    data = {"controls": controls, "construction_seed": 28, "seed": 19, "task_id": 7}
    first = session.request("construct", data)
    with pytest.raises(ValueError, match="Timed out: True"):
        session.request("construct", {**data, "construction_seed": 29, "max_seconds": 0.001})
    assert session.request("view", {})["view"]["episode_id"] == first["view"]["episode_id"]
    records = [json.loads(p.read_text()) for p in (tmp_path / ".attempts").rglob("*-result.json")]
    assert any(r["status"] == "FAILED" and r["result"]["timed_out"] for r in records)


def test_sampled_pair_preparation_and_verification_are_guarded(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from r0.modules.priors import SampledTopologyGraph

    session = ExplorerSession(tmp_path)
    controls = session.request("construction_examples", {})["examples"]["Cauchy causal graph"]
    controls["task"].update(rows=16, context_rows=8)
    controls["calibration_rows"] = 4
    original = session.request(
        "construct", {"controls": controls, "construction_seed": 28, "seed": 19, "task_id": 7}
    )

    def parent_pair(*args: object, **kwargs: object) -> object:
        raise AssertionError("sampled pair preparation/comparison must not run in parent")

    monkeypatch.setattr(SampledTopologyGraph, "with_task", parent_pair)
    paired = session.request("pair", {"intervention": "noise", "value": 0.8})
    assert paired["pair"]["card"]["control"]["after"] == 0.8
    with pytest.raises(ValueError, match="Timed out: True"):
        session.request("pair", {"intervention": "noise", "value": 0.7, "max_seconds": 0.001})
    assert session.request("view", {})["view"]["episode_id"] == original["view"]["episode_id"]
    failed = [
        json.loads(p.read_text()) for p in (tmp_path / ".pair-attempts").glob("*-failed.json")
    ]
    assert len(failed) == 1
    assert "Timed out: True" in failed[0]["error"]
    assert (tmp_path / ".pair-attempts" / f"{failed[0]['attempt']}-started.json").exists()
