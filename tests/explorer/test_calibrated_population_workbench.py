"""Fixed calibration controls preserve the original dataset and fit across Next."""

import json
from pathlib import Path

import pytest

from r0.modules.explorer import ExplorerSession


def test_calibrated_population_next_replay_and_pair(tmp_path: Path):
    session = ExplorerSession(tmp_path)
    controls = next(
        v
        for k, v in session.request("construction_examples", {})["examples"].items()
        if k.startswith("Population DAG: functional Conv")
    )
    cap = {"parameters": 16_000_000, "working_bytes": 4 * 1024**3}
    controls["request"]["limits"].update(cap)
    controls["request"]["base"]["limits"].update(cap)
    session.request(
        "construct",
        {
            "controls": controls,
            "construction_seed": "17",
            "causal_limits": cap,
            "seed": "31",
            "task_id": str(2**55 + 7),
        },
    )
    assert session.task is not None
    original = session.task.recipe_json
    fitted = session.request(
        "population_calibrate",
        {
            "calibration_rows": 32,
            "causal_limits": cap,
            "calibration_seed": "71",
            "calibration_task_id": str(2**55 + 73),
        },
    )
    source = fitted["view"]["recipe"]["recipe"]
    assert source["version"] == "calibrated-population-graph-v1"
    assert (
        json.loads(session.task.recipe_json)["recipe"]["original"] == json.loads(original)["recipe"]
    )
    fresh = session.request(
        "population_query",
        {
            "seed": "101",
            "task_id": str(2**55 + 103),
            "construction_seed": "107",
            "engine": "Philox",
            "causal_limits": cap,
        },
    )
    assert fresh["view"]["recipe"]["recipe"]["version"] == "calibrated-population-query-v1"
    assert (
        session.request("population_dag_replay", {"causal_limits": cap})["view"]["episode_id"]
        == fresh["view"]["episode_id"]
    )
    trace = session.request("graph", {"indices": [15, 0, 15], "causal_limits": cap})
    assert trace["graph"]["rows"] == [15, 0, 15]
    info = session.request("intervention_controls", {})
    assert info["calibration"]["query_fit"] is False
    assert info["calibration"]["fit"]["task_id"] == str(2**55 + 73)
    pair = session.request(
        "intervene",
        {
            "interventions_json": json.dumps(
                [
                    {
                        "version": "hard-causal-intervention-v1",
                        "node": 1,
                        "channels": [0],
                        "values": [2.0],
                        "category": None,
                        "location": "state_and_readout",
                    }
                ]
            ),
            "mode": "matched",
            "causal_limits": cap,
            "population": "1",
        },
    )
    assert pair["intervention"]
    assert session.request(
        "intervention_export",
        {"expected_intervention_id": pair["intervention"]["after_id"], "causal_limits": cap},
    )["data"]


def test_calibration_failure_retains_prior_state_and_failed_receipt(tmp_path: Path):
    session = ExplorerSession(tmp_path)
    controls = next(
        v
        for k, v in session.request("construction_examples", {})["examples"].items()
        if k.startswith("Population DAG: native linear")
    )
    session.request(
        "construct",
        {"controls": controls, "construction_seed": "17", "seed": "31", "task_id": "41"},
    )
    assert session.task is not None
    original = session.task
    with pytest.raises(ValueError, match="Generation failed"):
        session.request(
            "population_calibrate",
            {"calibration_rows": 1, "calibration_seed": "71", "calibration_task_id": "73"},
        )
    assert session.task is original
    receipts = [json.loads(p.read_text()) for p in (tmp_path / ".attempts").rglob("*-result.json")]
    assert any(r["status"] == "FAILED" for r in receipts)


def test_calibration_shared_notebook_callback(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    import sys
    from types import ModuleType
    from typing import Any

    from r0.modules.explorer import show_colab_explorer

    callbacks: dict[str, Any] = {}
    shown: list[str] = []
    output = ModuleType("google.colab.output")
    display = ModuleType("IPython.display")

    def register(name: str, callback: Any) -> None:
        callbacks[name] = callback

    def wrap(value: Any) -> dict[str, Any]:
        return {"application/json": value}

    monkeypatch.setattr(output, "register_callback", register, raising=False)
    monkeypatch.setattr(display, "JSON", wrap, raising=False)
    monkeypatch.setattr(display, "HTML", str, raising=False)
    monkeypatch.setattr(display, "display", shown.append, raising=False)
    monkeypatch.setitem(sys.modules, "google.colab.output", output)
    monkeypatch.setitem(sys.modules, "IPython.display", display)
    invoke = callbacks[show_colab_explorer(tmp_path)]
    assert 'id="population-calibrate"' in shown[0]
    controls = next(
        v
        for k, v in invoke("construction_examples", {})["application/json"]["examples"].items()
        if k.startswith("Population DAG: converted linear")
    )
    cap = {"parameters": 16_000_000, "working_bytes": 4 * 1024**3}
    controls["request"]["limits"].update(cap)
    controls["request"]["base"]["limits"].update(cap)
    invoke(
        "construct",
        {
            "controls": controls,
            "seed": "31",
            "task_id": "41",
            "construction_seed": "17",
            "causal_limits": cap,
        },
    )
    result = invoke(
        "population_calibrate",
        {
            "calibration_rows": 32,
            "calibration_seed": "71",
            "calibration_task_id": "73",
            "causal_limits": cap,
        },
    )["application/json"]
    assert "error" not in result, result
    assert result["view"]["recipe"]["recipe"]["version"] == "calibrated-population-graph-v1"
    opened = invoke("open", {"archive": result["archive"], "task_id": "41"})["application/json"]
    assert opened["recipe_json"] == result["recipe_json"]
    assert invoke("export", {})["application/json"]["data"]
