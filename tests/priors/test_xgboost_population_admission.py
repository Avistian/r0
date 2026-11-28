"""Complete outer population semantics reject before replay or backend work."""

import copy
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from gp_root_fixture import gp_root_recipe

from r0.modules import priors as p


@pytest.fixture(scope="module")
def graph():
    return p.fit_xgboost_population_graph(
        gp_root_recipe("functional").base,
        (
            (
                1,
                p.PopulationXGBoostPlan(
                    rows=8, input_width=2, output_width=1, estimators=2, depth=2
                ),
            ),
        ),
        population="support-only",
        seed=47,
        task_id=59,
        target_rng=np.random.default_rng(7),
        backend_rng=np.random.default_rng(11),
    )


@pytest.mark.parametrize(
    "change",
    [
        "version",
        "seed",
        "population",
        "node",
        "duplicate",
        "rows",
        "prefix",
        "fit-version",
        "raw-shape",
        "snapshot",
        "target",
        "caps",
    ],
)
def test_invalid_population_graph_metadata_precedes_restore(
    graph: p.XGBoostPopulationGraph, monkeypatch: pytest.MonkeyPatch, change: str
):
    raw: dict[str, Any] = asdict(graph)
    node = raw["fits"][0]
    fit = node["realization"]
    if change == "version":
        raw["version"] = "unknown"
    elif change == "seed":
        raw["fit_seed"] = True
    elif change == "population":
        raw["population"] = "other"
    elif change == "node":
        node["node"] = 0
    elif change == "duplicate":
        raw["fits"] = (node, node)
    elif change == "rows":
        node["row_ids"] = (0,) * 8
    elif change == "prefix":
        node["prefix_digest"] = "0" * 64
    elif change == "fit-version":
        fit["version"] = "unknown"
    elif change == "raw-shape":
        fit["raw"][0]["shape"] = (7, 1)
    elif change == "snapshot":
        fit["final_rng"][1]["engine"] = "unknown"
    elif change == "target":
        fit["targets"] = ((999.0,),) * 8
    else:
        raw["limits"]["working_bytes"] = 1

    def forbidden(*args: object, **kwargs: object):
        pytest.fail("outer population metadata initiated restore")

    monkeypatch.setattr(type(graph.fits[0].realization.initial_rng[0]), "restore", forbidden)
    with pytest.raises(ValueError, match="."):
        p.recipe_from_dict(raw)


def test_population_backend_failure_rolls_back_both_streams(monkeypatch: pytest.MonkeyPatch):
    import importlib

    backend = importlib.import_module("xgboost")
    streams = [np.random.default_rng(i) for i in (7, 11)]
    before = [copy.deepcopy(r.bit_generator.state) for r in streams]

    def fail(*args: object, **kwargs: object):
        raise RuntimeError("failed backend")

    monkeypatch.setattr(backend, "train", fail)
    with pytest.raises(RuntimeError, match="failed backend"):
        p.fit_xgboost_population_graph(
            gp_root_recipe("functional").base,
            ((1, p.PopulationXGBoostPlan(rows=8, input_width=2, output_width=1)),),
            population="support-only",
            seed=47,
            task_id=59,
            target_rng=streams[0],
            backend_rng=streams[1],
        )
    assert [r.bit_generator.state for r in streams] == before


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("seed", 48),
        ("count", 2),
        ("device", "cuda"),
        ("max_seconds", 0),
        ("max_bytes", 1),
        ("group_size", 0),
    ],
)
def test_population_run_header_rejects_before_decode(
    graph: p.XGBoostPopulationGraph, monkeypatch: pytest.MonkeyPatch, field: str, value: Any
):
    from r0.modules import generation as g

    document = dict(
        recipe=asdict(graph), seed=47, start=59, count=1, max_seconds=60, max_bytes=10_000_000
    )
    document[field] = value

    def forbidden(*args: object, **kwargs: object):
        pytest.fail("invalid run header restored source")

    monkeypatch.setattr(type(graph.fits[0].realization.initial_rng[0]), "restore", forbidden)
    with pytest.raises(ValueError, match="."):
        g.request_from_dict(document)


@pytest.mark.parametrize("change", ["valid", "identity", "execution", "targets"])
def test_population_producer_task_custody(
    graph: p.XGBoostPopulationGraph, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, change: str
):
    import json
    from dataclasses import replace

    from r0.modules import generation as g

    task = p.generate_task(graph, seed=47, task_id=59)
    raw = json.loads(task.recipe_json)
    if change == "identity":
        raw["seed"] = True
    elif change == "execution":
        raw["execution"]["python"] = ""
    elif change == "targets":
        task = replace(task, targets=task.targets + 1)
    task = replace(task, recipe_json=json.dumps(raw))

    def forbidden(*args: object, **kwargs: object):
        pytest.fail("invalid producer header restored source")

    if change in ("identity", "execution"):
        monkeypatch.setattr(type(graph.fits[0].realization.initial_rng[0]), "restore", forbidden)
    request = g.RunRequest(graph, seed=47, start=59, count=1, max_seconds=60, max_bytes=10_000_000)
    result = g.run_generation(request, tmp_path / "run", produce=lambda _: task)
    assert result.completed == (1 if change == "valid" else 0)


@pytest.mark.parametrize("change", ["missing", "boolean", "task"])
def test_population_inspection_rejects_outer_source_before_restore(
    graph: p.XGBoostPopulationGraph, monkeypatch: pytest.MonkeyPatch, change: str
):
    import json
    from dataclasses import replace

    task = p.generate_task(graph, seed=47, task_id=59)
    raw = json.loads(task.recipe_json)
    if change == "missing":
        del raw["recipe"]["population"]
    elif change == "boolean":
        raw["recipe"]["fit_seed"] = True
    else:
        raw["recipe"]["base"]["task"] = {}
    task = replace(task, recipe_json=json.dumps(raw))

    def forbidden(*args: object, **kwargs: object):
        pytest.fail("invalid inspection source restored streams")

    monkeypatch.setattr(type(graph.fits[0].realization.initial_rng[0]), "restore", forbidden)
    with pytest.raises(ValueError, match="."):
        p.explain_generation(task, [0])


@pytest.mark.parametrize("budget", ["working_bytes", "parameters"])
def test_population_fit_budget_precedes_draw_and_backend(
    budget: str, monkeypatch: pytest.MonkeyPatch
):
    import importlib
    from dataclasses import replace

    backend = importlib.import_module("xgboost")
    streams = [np.random.default_rng(i) for i in (7, 11)]
    before = [copy.deepcopy(r.bit_generator.state) for r in streams]

    def forbidden(*args: object, **kwargs: object):
        pytest.fail("inadmissible population reached backend")

    monkeypatch.setattr(backend, "train", forbidden)
    with pytest.raises(ValueError, match="."):
        p.fit_xgboost_population_graph(
            gp_root_recipe("functional").base,
            ((1, p.PopulationXGBoostPlan(rows=8, input_width=2, output_width=1)),),
            population="support-only",
            seed=47,
            task_id=59,
            target_rng=streams[0],
            backend_rng=streams[1],
            limits=replace(p.CausalLimits(), **{budget: 1}),
        )
    assert [r.bit_generator.state for r in streams] == before
