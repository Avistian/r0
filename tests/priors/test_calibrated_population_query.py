"""Fresh query populations retain the complete source and never fit new target constants."""

import json
import sys
from dataclasses import asdict
from pathlib import Path
from types import FrameType
from typing import Any, Literal

import numpy as np
import pytest

from r0.modules import priors as p


def _source(
    family: str = "functional",
    task_type: Literal["regression", "binary", "multiclass"] = "regression",
):
    width = 3 if task_type == "multiclass" else 1
    matrix = tuple(tuple(float(i == j) for j in range(width)) for i in range(width))
    limits = p.CausalLimits(parameters=16_000_000, working_bytes=4 * 1024**3)
    task = p.LinearRecipe(
        rows=16,
        context_rows=7,
        features=1,
        signal_features=1,
        active_features=1,
        task_type=task_type,
        classes={"regression": 0, "binary": 2, "multiclass": 3}[task_type],
    )
    nodes = (
        p.FunctionalCausalNode(
            (), p.FrozenLinearFunction(matrix), p.JointRootBase("gaussian", width)
        ),
        p.FunctionalCausalNode((0,), p.FrozenLinearFunction(matrix)),
    )
    if family == "functional":
        base = p.FunctionalCausalRecipe(
            task, nodes, ((0, 0),), tuple((1, j) for j in range(width)), limits=limits
        )
    elif family == "native":
        base = p.NativeCausalRecipe(
            task, tuple(p.NativeCausalNode(n) for n in nodes), ((0, 0),), (1, 0), limits=limits
        )
    elif family == "converted":
        base = p.ConvertedCausalRecipe(
            task,
            tuple(p.ConvertedCausalNode(n) for n in nodes),
            (p.ConvertedReadout(0, channel=0),),
            p.ConvertedReadout(1, channel=0),
            limits=limits,
        )
    else:
        base = p.NativeVectorRecipe(
            task,
            tuple(p.NativeVectorNode(n, (p.VariableDomain(),)) for n in nodes),
            ((0, 0),),
            (1, 0),
            limits=limits,
        )
    forest = p.RecursiveForestPlan(
        input_width=width,
        intervals=((-1.0, 1.0),) * width,
        estimators=1,
        min_depth=1,
        max_depth=1,
        lifecycle="fresh-forest-per-row",
        populations=16,
        source="r0-extension-v1",
        leaf_mean=(0.1,) * width,
        leaf_scale=(1.0,) * width,
    )
    original = p.realize_recursive_population_graph(
        p.RecursivePopulationGraphRequest(
            base, (p.RecursiveNodePopulationPlan(1, forest),), limits=limits
        ),
        seed=37,
        task_id=2**55 + 9,
    )
    return p.calibrate_population_graph(
        original, calibration_rows=32, calibration_seed=71, calibration_task_id=73
    )


@pytest.mark.parametrize("engine", ["PCG64", "PCG64DXSM", "Philox", "SFC64", "MT19937"])
def test_fresh_query_keeps_fit_and_full_lineage(engine: str):
    source = _source()
    snapshot = asdict(source)
    calls: list[tuple[str, Any, Any]] = []

    def capture(frame: FrameType, event: str, arg: object):
        if event == "call" and frame.f_code is p.calibrate_population_graph.__code__:
            calls.append(("fit", None, None))
        if event == "call" and frame.f_code is p.realize_recursive_population_graph.__code__:
            calls.append(("realize", frame.f_locals["seed"], frame.f_locals["task_id"]))

    previous = sys.getprofile()
    sys.setprofile(capture)
    try:
        query = p.realize_calibrated_population_query(
            source, seed=101, task_id=2**55 + 11, engine=engine
        )
    finally:
        sys.setprofile(previous)
    assert calls == [("realize", 101, 2**55 + 11)]
    assert asdict(source) == snapshot
    assert asdict(query.source) == snapshot
    assert query.calibration == source.calibration
    assert query.calibration_population == source.calibration_population
    task = p.generate_task(query, seed=101, task_id=2**55 + 11)
    raw = p.generate_task(query.original, seed=101, task_id=2**55 + 11)
    np.testing.assert_allclose(
        task.targets,
        (raw.scores[:, 0].numpy() - source.calibration.center[0]) / source.calibration.scale[0],
    )
    restored = p.recipe_from_dict(json.loads(json.dumps(asdict(query))), limits=source.limits)
    replay = p.generate_task(restored, seed=101, task_id=2**55 + 11)
    assert replay.recipe_json == task.recipe_json
    next_query = p.realize_calibrated_population_query(
        query, seed=103, task_id=2**55 + 13, engine=engine
    )
    assert asdict(next_query.source) == asdict(query)
    assert next_query.calibration == source.calibration


@pytest.fixture(scope="module")
def query() -> p.CalibratedPopulationQuery:
    return p.realize_calibrated_population_query(_source(), seed=101, task_id=2**55 + 11)


def test_query_trace_transport_and_intervention(query: p.CalibratedPopulationQuery, tmp_path: Path):
    from r0.modules import generation as g

    task = p.generate_task(query, seed=101, task_id=2**55 + 11)
    trace = p.explain_generation(task, [15, 0, 15], limits=query.limits)
    assert trace["target_max_error"] == 0
    request = g.request_from_dict(
        asdict(
            g.RunRequest(
                query, 101, 1, 120, 64 * 1024**2, start=2**55 + 11, causal_limits=query.limits
            )
        )
    )
    result = g.run_generation(request, tmp_path / "query")
    assert result.completed == 1
    change = p.CausalInterventionRecipe(
        query.original.request.base, (p.HardIntervention(1, (0,), (2.5,)),)
    )
    pair = p.generate_calibrated_population_pair(query, change)
    assert pair.before.recipe_json == task.recipe_json
    np.testing.assert_allclose(
        pair.after.targets, (2.5 - query.calibration.center[0]) / query.calibration.scale[0]
    )
    assert (
        p.explain_generation(pair.after, [15, 0, 15], limits=query.limits)["target_max_error"] == 0
    )
    g.save_causal_pair(tmp_path / "pair", pair, max_bytes=64 * 1024**2, limits=query.limits)
    loaded = g.load_causal_pair(tmp_path / "pair", limits=query.limits)
    assert loaded.after.recipe_json == pair.after.recipe_json


@pytest.mark.parametrize("family", ["native", "converted", "vector"])
def test_typed_query_preserves_direct_continuous_semantics(family: str):
    source = _source(family)
    query = p.realize_calibrated_population_query(source, seed=109, task_id=113)
    task = p.generate_task(query, seed=109, task_id=113)
    original = p.generate_task(query.original, seed=109, task_id=113)
    np.testing.assert_allclose(
        task.targets,
        (original.targets.numpy() - source.calibration.center[0]) / source.calibration.scale[0],
    )
    assert task.feature_schema == original.feature_schema
    assert task.target_origin == original.target_origin
    assert task.classes == original.classes
    trace = p.explain_generation(task, [15, 0, 15], limits=query.limits)
    assert trace["target_max_error"] == 0


@pytest.mark.parametrize("task_type", ["binary", "multiclass"])
def test_query_classification_uses_fixed_scores(task_type: Literal["binary", "multiclass"]):
    source = _source(task_type=task_type)
    query = p.realize_calibrated_population_query(source, seed=109, task_id=113)
    task = p.generate_task(query, seed=109, task_id=113)
    original = p.generate_task(query.original, seed=109, task_id=113)
    scores = (original.scores.numpy() - np.array(source.calibration.center)) / np.array(
        source.calibration.scale
    )
    if task_type == "binary":
        positive = 1 / (1 + np.exp(-scores[:, 0] * source.task.separation))
        expected = np.column_stack((1 - positive, positive))
    else:
        logits = scores * source.task.separation
        exp = np.exp(logits - logits.max(axis=1, keepdims=True))
        expected = exp / exp.sum(axis=1, keepdims=True)
    assert task.probabilities is not None
    np.testing.assert_allclose(task.probabilities, expected, atol=1e-14)
    assert p.explain_generation(task, [15, 0, 15], limits=query.limits)["target_max_error"] == 0


@pytest.mark.parametrize(
    "mutation", ["law", "identity", "engine", "source", "construction-seed", "budget"]
)
def test_query_lineage_admission_precedes_rng(query: p.CalibratedPopulationQuery, mutation: str):
    from dataclasses import replace
    from unittest.mock import patch

    raw = json.loads(json.dumps(asdict(query)))
    limits = query.limits
    if mutation == "law":
        raw["original"]["request"]["base"]["nodes"][0]["root"]["family"] = "uniform"
    elif mutation == "identity":
        raw["original"]["seed"] = raw["source"]["original"]["seed"]
        raw["original"]["task_id"] = raw["source"]["original"]["task_id"]
    elif mutation == "engine":
        raw["engine"] = "unknown"
    elif mutation == "construction-seed":
        raw["construction_seed"] = True
    elif mutation == "source":
        raw["source"]["calibration_population"]["seed"] = True
    else:
        limits = replace(limits, parameters=1)
    cls = type(query.original.bindings[0].sample.initial)
    with (
        patch.object(cls, "restore", side_effect=AssertionError("lineage restored RNG")),
        patch.object(np.random, "default_rng", side_effect=AssertionError("lineage created RNG")),
        pytest.raises(ValueError, match="query|calibrat|population|limit|metadata"),
    ):
        p.recipe_from_dict(raw, limits=limits)


@pytest.mark.parametrize("path", ["generation", "inspection", "request"])
def test_query_outer_identity_rejects_before_restore(query: p.CalibratedPopulationQuery, path: str):
    from dataclasses import replace
    from unittest.mock import patch

    from r0.modules import generation as g

    task = p.generate_task(query, seed=101, task_id=2**55 + 11)
    record = json.loads(task.recipe_json)
    record["seed"] = 102
    cls = type(query.original.bindings[0].sample.initial)

    def invoke():
        if path == "generation":
            p.generate_task(query, seed=102, task_id=2**55 + 11)
        elif path == "inspection":
            p.explain_generation(
                replace(task, recipe_json=json.dumps(record)), [0], limits=query.limits
            )
        else:
            g.request_from_dict(
                {
                    "recipe": asdict(query),
                    "seed": 102,
                    "start": 2**55 + 11,
                    "count": 1,
                    "max_seconds": 120,
                    "max_bytes": 64 * 1024**2,
                    "causal_limits": asdict(query.limits),
                }
            )

    with (
        patch.object(cls, "restore", side_effect=AssertionError("identity restored RNG")),
        pytest.raises(ValueError, match="identity"),
    ):
        invoke()


def test_next_query_cannot_reuse_any_ancestor_identity(query: p.CalibratedPopulationQuery):
    from unittest.mock import patch

    root, _ = query.lineage()
    cls = type(query.original.bindings[0].sample.initial)
    with (
        patch.object(cls, "restore", side_effect=AssertionError("repeated identity restored RNG")),
        pytest.raises(ValueError, match="identity"),
    ):
        p.realize_calibrated_population_query(
            query, seed=root.original.seed, task_id=root.original.task_id
        )


def test_query_construction_seed_is_owned_and_used(query: p.CalibratedPopulationQuery):
    source = query.source
    first = p.realize_calibrated_population_query(
        source, seed=127, task_id=131, construction_seed=137
    )
    second = p.realize_calibrated_population_query(
        source, seed=127, task_id=131, construction_seed=139
    )
    assert first.construction_seed == 137
    assert second.construction_seed == 139
    assert first.original.bindings[0].sample.initial != second.original.bindings[0].sample.initial
    assert first.original.bindings[0].sample.forests != second.original.bindings[0].sample.forests
    np.testing.assert_array_equal(
        first.original.bindings[0].inputs, second.original.bindings[0].inputs
    )
    assert first.calibration == second.calibration
    assert first.calibration_population == second.calibration_population


def test_query_heterogeneous_parent_encoding():
    limits = p.CausalLimits(parameters=16_000_000, working_bytes=4 * 1024**3)
    task = p.LinearRecipe(rows=16, context_rows=7, features=2, signal_features=2, active_features=2)
    root = p.NativeVectorNode(
        p.FunctionalCausalNode(
            (),
            p.FrozenLinearFunction(((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))),
            p.JointRootBase("gaussian", 3),
        ),
        (p.VariableDomain("binary", ("no", "yes"), "softmax"), p.VariableDomain()),
    )
    target = p.NativeVectorNode(
        p.FunctionalCausalNode((0,), p.FrozenLinearFunction(((1.0, 1.0, 1.0),))),
        (p.VariableDomain(),),
    )
    base = p.NativeVectorRecipe(task, (root, target), ((0, 0), (0, 1)), (1, 0), limits=limits)
    plan = p.RecursiveForestPlan(
        input_width=3,
        intervals=((-1.0, 1.0),) * 3,
        estimators=1,
        min_depth=1,
        max_depth=1,
        lifecycle="fresh-forest-per-row",
        populations=16,
    )
    original = p.realize_recursive_population_graph(
        p.RecursivePopulationGraphRequest(
            base, (p.RecursiveNodePopulationPlan(1, plan),), limits=limits
        ),
        seed=37,
        task_id=41,
    )
    source = p.calibrate_population_graph(
        original, calibration_rows=32, calibration_seed=71, calibration_task_id=73
    )
    query = p.realize_calibrated_population_query(source, seed=101, task_id=103)
    parents = np.array(query.original.bindings[0].inputs)
    np.testing.assert_array_equal(parents[:, :2].sum(axis=1), np.ones(16))
    assert np.all((parents[:, :2] == 0) | (parents[:, :2] == 1))
    generated = p.generate_task(query, seed=101, task_id=103)
    assert generated.feature_schema[0].kind == "binary"
    assert p.explain_generation(generated, [15, 0, 15], limits=limits)["target_max_error"] == 0
