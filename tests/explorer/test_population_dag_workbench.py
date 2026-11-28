"""Population DAG workbench uses guarded construction and retained original rows."""

import json
from pathlib import Path
from typing import Any

import pytest

from r0.modules import generation as g
from r0.modules import priors as p
from r0.modules.explorer import ExplorerSession


def test_population_catalog_builds_all_families_and_original_row_traces(tmp_path: Path):
    catalog = ExplorerSession(tmp_path).request("construction_examples", {})["examples"]
    entries = {k: v for k, v in catalog.items() if k.startswith("Population DAG:")}
    assert len(entries) >= 8
    families: set[str] = set()
    for controls in entries.values():
        recipe = g.GraphConstruction(json.dumps(controls), 17).build(
            p.CausalLimits(), episode_seed=31, task_id=2**55 + 7
        )
        assert isinstance(recipe, p.RecursivePopulationGraph)
        families.add(recipe.request.base.version)
        task = p.generate_task(recipe, seed=31, task_id=2**55 + 7)
        trace = p.explain_generation(task, [15, 0, 15])
        assert trace["rows"] == [15, 0, 15]
        assert any("population" in n["id"] for n in trace["nodes"])
    assert families == {
        "functional-causal-v1",
        "native-causal-v1",
        "native-vector-causal-v1",
        "converted-causal-v1",
    }


@pytest.mark.parametrize("family", ["functional", "native", "vector", "converted"])
@pytest.mark.parametrize("mode", ["matched", "population"])
def test_guarded_population_session_replay_pair_and_export(tmp_path: Path, family: str, mode: str):
    session = ExplorerSession(tmp_path)
    controls = next(
        v
        for k, v in session.request("construction_examples", {})["examples"].items()
        if k.startswith(f"Population DAG: {family} Conv")
    )
    first = session.request(
        "construct",
        {"controls": controls, "construction_seed": "17", "seed": "31", "task_id": str(2**55 + 7)},
    )
    repeated = session.request("population_dag_replay", {})
    assert repeated["view"]["episode_id"] == first["view"]["episode_id"]
    info = session.request("intervention_controls", {})
    assert info["population_dag"]
    assert info["nodes"][1]["population_forest"]["lifecycle"] == "fresh-forest-per-row"
    change = {
        "version": "hard-causal-intervention-v1",
        "node": 1,
        "channels": [0],
        "values": [2.0],
        "category": None,
        "location": "state_and_readout",
    }
    if family == "native":
        change["location"] = "state"
    elif family == "vector":
        change = {
            "version": "native-vector-hard-intervention-v1",
            "node": 1,
            "variable": 0,
            "value": 2.0,
            "category": None,
        }
    pair = session.request(
        "intervene",
        {"interventions_json": json.dumps([change]), "mode": mode, "population": "1"},
    )
    assert pair["intervention"]
    trace = session.request(
        "intervention_graph",
        {"indices": [15, 0, 15], "expected_intervention_id": pair["intervention"]["after_id"]},
    )
    target = next(n for n in trace["after"]["nodes"] if n["id"] == "target")
    assert target["values"] == [[6.0], [6.0], [6.0]]
    exported = session.request(
        "intervention_export", {"expected_intervention_id": pair["intervention"]["after_id"]}
    )
    assert exported["data"]


def test_invalid_complete_construction_rejects_before_realization(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    from copy import deepcopy

    catalog = ExplorerSession(tmp_path).request("construction_examples", {})["examples"]
    controls = next(
        v for k, v in catalog.items() if k.startswith("Population DAG: functional linear")
    )

    def forbidden(*args: object, **kwargs: object):
        raise AssertionError("population realization started before complete admission")

    monkeypatch.setattr(p, "realize_recursive_population_graph", forbidden)
    invalid: list[dict[str, Any]] = []
    for key, value in (
        ("populations", 15),
        ("max_depth", 11),
        ("lifecycle", "frozen-function"),
        ("input_width", 3),
    ):
        bad = deepcopy(controls)
        bad["request"]["replacements"][0]["forest"][key] = value
        invalid.append(bad)
    bad = deepcopy(controls)
    bad["request"]["replacements"][0]["node"] = 99
    invalid.append(bad)
    bad = deepcopy(controls)
    bad["engine"] = "bad"
    invalid.append(bad)
    for bad in invalid:
        with pytest.raises(
            ValueError,
            match="population|recursive|width|engine|limit|intervention|hard|channel|location|finite|node|cap|budget",
        ):
            g.GraphConstruction(json.dumps(bad), 17).build(
                p.CausalLimits(), episode_seed=31, task_id=7
            )
    with pytest.raises(
        ValueError,
        match="population|recursive|width|engine|limit|intervention|hard|channel|location|finite|node|cap|budget",
    ):
        g.GraphConstruction(json.dumps(controls), 17).build(
            p.CausalLimits(working_bytes=65536), episode_seed=31, task_id=7
        )


def test_population_change_metadata_rejects_before_snapshot_restore(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    from copy import deepcopy
    from dataclasses import asdict

    import numpy as np

    controls = next(
        v
        for k, v in ExplorerSession(tmp_path)
        .request("construction_examples", {})["examples"]
        .items()
        if k.startswith("Population DAG: functional Conv")
    )
    conv = p.sample_feature_conv1d(
        p.FeatureConv1DRequest(2, 1, (p.Conv1DLayerPrior((p.Conv1DGeometry(1, 1),)),)),
        named_rng=np.random.default_rng(3),
        raw_rng=np.random.default_rng(5),
    )
    controls["request"]["base"]["nodes"][0]["function"] = asdict(conv)
    base = g.GraphConstruction(json.dumps(controls), 17).build(
        p.CausalLimits(), episode_seed=31, task_id=7
    )
    assert isinstance(base, p.RecursivePopulationGraph)
    record = asdict(base)

    def forbidden(*args: object, **kwargs: object):
        raise AssertionError("snapshot restored before change semantic admission")

    monkeypatch.setattr(type(base.bindings[0].sample.initial), "restore", forbidden)
    change = {
        "version": "hard-causal-intervention-v1",
        "node": 1,
        "channels": [0],
        "values": [2.0],
        "category": None,
        "location": "state",
    }
    for mode, population in (("bad", 1), ("population", True)):
        with pytest.raises(
            ValueError,
            match="population|recursive|width|engine|limit|intervention|hard|channel|location|finite|node|cap|budget",
        ):
            p.construct_population_graph_change(
                record, [change], mode=mode, population=population, limits=p.CausalLimits()
            )
    for field, value in (
        ("node", 99),
        ("channels", [99]),
        ("location", "bad"),
        ("values", [float("inf")]),
    ):
        bad = deepcopy(change)
        bad[field] = value
        with pytest.raises(
            ValueError,
            match="population|recursive|width|engine|limit|intervention|hard|channel|location|finite|node|cap|budget",
        ):
            p.construct_population_graph_change(
                record, [bad], mode="matched", population=1, limits=p.CausalLimits()
            )
    with pytest.raises(
        ValueError,
        match="population|recursive|width|engine|limit|intervention|hard|channel|location|finite|node|cap|budget",
    ):
        p.construct_population_graph_change(
            record,
            [change],
            mode="matched",
            population=1,
            limits=p.CausalLimits(working_bytes=32 * 1024**2),
        )
