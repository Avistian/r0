"""A reopened calibrated comparison stays a retained archive, never a new query."""

import json
from dataclasses import asdict
from pathlib import Path

import pytest

from r0.modules import generation as g
from r0.modules import priors as p
from r0.modules.explorer import ExplorerSession


@pytest.mark.parametrize("mode", ["matched", "population"])
def test_reopened_calibrated_pair_replay_details_and_unavailable_next(tmp_path: Path, mode: str):
    cap = p.CausalLimits(parameters=16_000_000, working_bytes=4 * 1024**3)
    session = ExplorerSession(tmp_path)
    controls = next(
        v
        for k, v in session.request("construction_examples", {})["examples"].items()
        if k.startswith("Population DAG: functional linear")
    )
    controls["request"]["limits"] = asdict(cap)
    controls["request"]["base"]["limits"] = asdict(cap)
    original = g.GraphConstruction(json.dumps(controls), 17).build(
        cap, episode_seed=31, task_id=2**55 + 7
    )
    assert isinstance(original, p.RecursivePopulationGraph)
    fit = p.calibrate_population_graph(
        original, calibration_rows=32, calibration_seed=71, calibration_task_id=2**55 + 73
    )
    changed = p.intervene_calibrated_population(
        fit,
        p.CausalInterventionRecipe(
            original.request.base,
            (p.HardIntervention(1, (0,), (2.0,), location="state_and_readout"),),
            mode=mode,
            population=7,
        ),
    )
    task = p.generate_task(changed, seed=31, task_id=2**55 + 7)
    archive = tmp_path / "after.zip"
    g.save_group(archive, [(2**55 + 7, task)], max_bytes=256 * 1024**2)
    opened = session.request("open", {"archive": str(archive), "task_id": str(2**55 + 7)})
    assert opened["recipe_json"] == task.recipe_json
    replayed = session.request("population_dag_replay", {"causal_limits": asdict(cap)})
    assert json.loads(replayed["recipe_json"])["recipe"] == json.loads(task.recipe_json)["recipe"]
    assert replayed["view"]["episode_id"] == opened["view"]["episode_id"]
    info = session.request("population_dag_controls", {})
    assert info["calibration"]["query_fit"] is False
    assert info["calibration"]["fit"]["task_id"] == str(2**55 + 73)
    assert info["retained_comparison"] is True
    with pytest.raises(ValueError, match="retained comparison"):
        session.request("population_query", {"seed": "101", "task_id": "103"})
    assert session.task is not None
    assert session.task.recipe_json == replayed["recipe_json"]
    assert session.request("export", {})["data"]
