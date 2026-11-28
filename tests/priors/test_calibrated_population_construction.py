"""Raw guarded controls reject known malformed metadata before source replay."""

from dataclasses import asdict
from pathlib import Path
from typing import Any
from unittest.mock import patch

import numpy as np
import pytest

from r0.modules import priors as p


@pytest.fixture
def source():
    cap = p.CausalLimits(parameters=16_000_000, working_bytes=4 * 1024**3)
    task = p.LinearRecipe(rows=16, context_rows=8, features=1, signal_features=1, active_features=1)
    nodes = (
        p.FunctionalCausalNode(
            (), p.FrozenLinearFunction(((1.0,),)), p.JointRootBase("gaussian", 1)
        ),
        p.FunctionalCausalNode((0,), p.FrozenLinearFunction(((1.0,),))),
    )
    base = p.FunctionalCausalRecipe(task, nodes, ((0, 0),), ((1, 0),), limits=cap)
    plan = p.RecursiveForestPlan(
        input_width=1,
        intervals=((-1.0, 1.0),),
        estimators=1,
        min_depth=1,
        max_depth=2,
        lifecycle="fresh-forest-per-row",
        populations=16,
    )
    return p.realize_recursive_population_graph(
        p.RecursivePopulationGraphRequest(
            base, (p.RecursiveNodePopulationPlan(1, plan),), limits=cap
        ),
        seed=31,
        task_id=41,
    )


@pytest.mark.parametrize(
    "mutation", ["rows", "seed", "identity", "extra", "owner", "budget", "snapshot"]
)
def test_raw_fit_controls_precede_restore(source: p.RecursivePopulationGraph, mutation: str):
    raw: dict[str, Any] = {
        "topology": "calibrate-population",
        "source": asdict(source),
        "calibration_rows": 32,
        "calibration_seed": 71,
        "calibration_task_id": 73,
    }
    seed = 31
    cap = source.limits
    if mutation == "rows":
        raw["calibration_rows"] = 1
    elif mutation == "seed":
        raw["calibration_seed"] = True
    elif mutation == "identity":
        raw["calibration_seed"] = 31
        raw["calibration_task_id"] = 41
    elif mutation == "extra":
        raw["unexpected"] = 1
    elif mutation == "owner":
        seed = 32
    elif mutation == "budget":
        cap = p.CausalLimits(parameters=1)
    else:
        raw["source"]["bindings"][0]["sample"]["initial"]["version"] = "bad"
    cls = type(source.bindings[0].sample.initial)
    with (
        patch.object(cls, "restore", side_effect=AssertionError("restored source")),
        patch.object(np.random, "default_rng", side_effect=AssertionError("created stream")),
        pytest.raises(ValueError, match="calibration|population|identity|limit|snapshot|metadata"),
    ):
        p.construct_calibrated_population(
            raw, seed=seed, task_id=41, construction_seed=71, limits=cap
        )


def test_explicit_fit_query_matches_public_fixed_formula(source: p.RecursivePopulationGraph):
    fitted = p.construct_calibrated_population(
        {
            "topology": "calibrate-population",
            "source": asdict(source),
            "calibration_rows": 32,
            "calibration_seed": 71,
            "calibration_task_id": 73,
        },
        seed=31,
        task_id=41,
        construction_seed=71,
        limits=source.limits,
    )
    assert isinstance(fitted, p.CalibratedPopulationGraph)
    query = p.construct_calibrated_population(
        {"topology": "calibrated-population-query", "source": asdict(fitted), "engine": "Philox"},
        seed=101,
        task_id=103,
        construction_seed=107,
        limits=source.limits,
    )
    assert isinstance(query, p.CalibratedPopulationQuery)
    assert asdict(query.source) == asdict(fitted)
    generated = p.generate_task(query, seed=101, task_id=103)
    raw = p.generate_task(query.original, seed=101, task_id=103)
    fit = p.generate_task(fitted.calibration_population, seed=71, task_id=73)
    values = fit.scores.numpy()
    np.testing.assert_allclose(
        generated.targets.numpy(),
        ((raw.scores.numpy() - values.mean(0)) / values.std(0)).ravel(),
        atol=1e-13,
    )


@pytest.mark.parametrize("family", ["native", "vector", "converted"])
def test_guarded_construction_preserves_typed_family(tmp_path: Path, family: str):
    import json

    from r0.modules import generation as g
    from r0.modules.explorer import ExplorerSession, inspect_task

    controls = next(
        v
        for k, v in ExplorerSession(tmp_path)
        .request("construction_examples", {})["examples"]
        .items()
        if k.startswith(f"Population DAG: {family} Conv")
    )
    cap = p.CausalLimits(parameters=16_000_000, working_bytes=4 * 1024**3)
    controls["request"]["limits"] = asdict(cap)
    controls["request"]["base"]["limits"] = asdict(cap)
    original = g.GraphConstruction(json.dumps(controls), 17).build(cap, episode_seed=31, task_id=41)
    assert isinstance(original, p.RecursivePopulationGraph)
    fit = g.GraphConstruction(
        json.dumps(
            {
                "topology": "calibrate-population",
                "source": asdict(original),
                "calibration_rows": 32,
                "calibration_seed": 71,
                "calibration_task_id": 73,
            }
        ),
        71,
    ).build(cap, episode_seed=31, task_id=41)
    assert isinstance(fit, p.CalibratedPopulationGraph)
    query = g.GraphConstruction(
        json.dumps(
            {"topology": "calibrated-population-query", "source": asdict(fit), "engine": "SFC64"}
        ),
        107,
    ).build(cap, episode_seed=101, task_id=103)
    assert isinstance(query, p.CalibratedPopulationQuery)
    raw = p.generate_task(query.original, seed=101, task_id=103)
    actual = p.generate_task(query, seed=101, task_id=103)
    reference = p.generate_task(fit.calibration_population, seed=71, task_id=73).scores.numpy()
    np.testing.assert_allclose(
        actual.targets.numpy(),
        ((raw.scores.numpy() - reference.mean(0)) / reference.std(0)).ravel(),
        atol=1e-13,
    )
    assert actual.feature_schema == raw.feature_schema
    assert inspect_task(actual)["target_origin"] == actual.target_origin


def test_guarded_next_admits_prospective_wrapper_before_decode(source: p.RecursivePopulationGraph):
    current: p.CalibratedPopulationGraph | p.CalibratedPopulationQuery = (
        p.calibrate_population_graph(
            source, calibration_rows=32, calibration_seed=71, calibration_task_id=73
        )
    )
    for depth in range(1, 13):
        current = p.realize_calibrated_population_query(current, seed=101, task_id=103 + depth)
    controls = {
        "topology": "calibrated-population-query",
        "source": asdict(current),
        "engine": "PCG64",
    }
    cls = type(source.bindings[0].sample.initial)
    with (
        patch.object(
            cls, "restore", side_effect=AssertionError("prospective query restored source")
        ),
        pytest.raises(ValueError, match="traversal|depth"),
    ):
        p.construct_calibrated_population(
            controls, seed=101, task_id=117, construction_seed=107, limits=source.limits
        )
