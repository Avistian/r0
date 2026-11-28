"""Actual ancestor inputs, not iid replacements, determine population XGBoost fits."""

import importlib
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pytest

from r0.modules import priors as p


def test_actual_population_fit_matches_native_histogram_model():
    x = np.array([[i, (-1.0) ** i] for i in range(8)], dtype=np.float64)
    plan = p.PopulationXGBoostPlan(rows=8, input_width=2, output_width=2, estimators=2, depth=2)
    fit = p.fit_population_xgboost(
        plan,
        inputs=x,
        population="support-only",
        target_rng=np.random.default_rng(7),
        backend_rng=np.random.default_rng(11),
    )
    y = np.random.default_rng(7).standard_normal((8, 2))
    seed = int(np.random.default_rng(11).integers(0, 2**31, size=1)[0])
    np.testing.assert_array_equal(fit.inputs, x)
    np.testing.assert_array_equal(fit.targets, y)
    backend = importlib.import_module("xgboost")
    model = backend.train(
        plan.backend_parameters(seed), backend.DMatrix(x, label=y), num_boost_round=2
    )
    assert fit.model_json == bytes(model.save_raw(raw_format="json")).decode("ascii")
    np.testing.assert_allclose(
        p.evaluate_xgboost_vector(fit.function, x), model.predict(backend.DMatrix(x)), atol=1e-6
    )
    assert p.population_xgboost_fit_from_dict(asdict(fit)) == fit


@pytest.mark.parametrize("family", ["functional", "native", "vector", "converted"])
@pytest.mark.parametrize(
    "population", ["support-only", "batch-refit-reference", "independent-ancestors"]
)
def test_population_graph_fit_uses_actual_context_parent_rows(family: str, population: str):
    from gp_root_fixture import gp_root_recipe

    base = gp_root_recipe(family).base
    plan = p.PopulationXGBoostPlan(
        rows=8 if population == "support-only" else 16,
        input_width=2,
        output_width=1,
        estimators=2,
        depth=2,
    )
    graph = p.fit_xgboost_population_graph(
        base,
        ((1, plan),),
        population=population,
        seed=47,
        task_id=59,
        target_rng=np.random.default_rng(7),
        backend_rng=np.random.default_rng(11),
    )
    task = p.generate_task(
        graph, seed=48 if population == "independent-ancestors" else 47, task_id=59
    )
    fit = graph.fits[0]
    if population == "support-only":
        assert tuple(map(int, task.context_indices)) == fit.row_ids
    fit_task = p.generate_task(base, seed=47, task_id=59)
    np.testing.assert_array_equal(fit.realization.inputs, fit_task.inputs[list(fit.row_ids)])
    expected = p.evaluate_xgboost_vector(fit.realization.function, task.inputs.numpy())[:, 0]
    np.testing.assert_allclose(task.targets, expected, atol=1e-12)
    assert p.recipe_from_dict(asdict(graph)) == graph


def test_population_graph_zip_replay_and_fit_scopes(tmp_path: Path):
    from gp_root_fixture import gp_root_recipe

    from r0.modules import generation as g

    graph = p.fit_xgboost_population_graph(
        gp_root_recipe("converted").base,
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
        task_id=2**55 + 9,
        target_rng=np.random.default_rng(7),
        backend_rng=np.random.default_rng(11),
    )
    task = p.generate_task(graph, seed=47, task_id=2**55 + 9)
    path = tmp_path / "population.zip"
    g.save_group(path, [(2**55 + 9, task)], max_bytes=10_000_000)
    saved = next(g.load_group(path))
    assert p.validate_xgboost_population_task(saved, limits=p.CausalLimits()) == graph
    trace = p.explain_generation(saved, [15, 1, 15])
    assert trace["rows"] == [15, 1, 15]
    nodes = {n["id"]: n for n in trace["nodes"]}
    assert all(parent in nodes for node in nodes.values() for parent in node["parents"])
    assert nodes["fit-1-inputs"]["parameters"]["row_labels"] == list(graph.fits[0].row_ids)
    assert nodes["fit-1-inputs"]["parameters"]["scope"] == "fit-population"


def test_sequential_population_fits_retain_realized_upstream_model_inputs():
    from dataclasses import replace

    from gp_root_fixture import gp_root_recipe

    base = gp_root_recipe("functional").base
    assert isinstance(base, p.FunctionalCausalRecipe)
    base = replace(
        base,
        nodes=(
            base.nodes[0],
            p.FunctionalCausalNode((0,), p.FrozenLinearFunction(((1.0, 0.0), (0.0, 1.0)))),
            p.FunctionalCausalNode((1,), p.FrozenLinearFunction(((2.0, -1.0),))),
        ),
        target=((2, 0),),
    )
    plans = tuple(
        (
            node,
            p.PopulationXGBoostPlan(
                rows=8, input_width=2, output_width=width, estimators=2, depth=2
            ),
        )
        for node, width in ((1, 2), (2, 1))
    )
    graph = p.fit_xgboost_population_graph(
        base,
        plans,
        population="support-only",
        seed=47,
        task_id=59,
        target_rng=np.random.default_rng(7),
        backend_rng=np.random.default_rng(11),
    )
    first, second = graph.fits
    upstream = p.evaluate_xgboost_vector(
        first.realization.function, np.asarray(first.realization.inputs)
    )
    np.testing.assert_array_equal(second.realization.inputs, upstream)
    assert first.realization.final_rng == second.realization.initial_rng
    assert p.recipe_from_dict(asdict(graph)) == graph


def test_conv_parent_population_xgboost_linear_retains_detailed_trace():
    conv = p.FrozenFeatureConv1D(
        2, 1, (p.FeatureConv1DLayer(p.Conv1DGeometry(1, 2), (((2.0, -1.0),),), (0.5,)),)
    )
    base = p.FunctionalCausalRecipe(
        p.LinearRecipe(rows=16, context_rows=8, features=1, signal_features=1, active_features=1),
        (
            p.FunctionalCausalNode((), conv, p.JointRootBase("gaussian", 2)),
            p.FunctionalCausalNode((0,), p.FrozenLinearFunction(((1.0,),))),
            p.FunctionalCausalNode((1,), p.FrozenLinearFunction(((3.0,),))),
        ),
        ((0, 0),),
        ((2, 0),),
    )
    graph = p.fit_xgboost_population_graph(
        base,
        (
            (
                1,
                p.PopulationXGBoostPlan(
                    rows=8, input_width=1, output_width=1, estimators=2, depth=2
                ),
            ),
        ),
        population="support-only",
        seed=47,
        task_id=59,
        target_rng=np.random.default_rng(7),
        backend_rng=np.random.default_rng(11),
    )
    task = p.generate_task(graph, seed=47, task_id=59)
    trace = p.explain_generation(task, [15, 2, 15])
    nodes = {node["id"]: node for node in trace["nodes"]}
    raw = np.asarray(nodes["root-0"]["values"])
    affine = np.asarray(nodes["convolution-0-layer-0-affine"]["values"])[:, 0]
    np.testing.assert_allclose(affine, 2 * raw[:, 0] - raw[:, 1] + 0.5, atol=1e-15)
    np.testing.assert_array_equal(
        graph.fits[0].realization.inputs, task.inputs[task.context_indices]
    )
    fit = graph.fits[0].realization.function
    expected = 3 * p.evaluate_xgboost_vector(fit, task.inputs.numpy())[:, 0].astype(np.float64)
    np.testing.assert_array_equal(task.targets, expected)
    assert all(parent in nodes for node in nodes.values() for parent in node["parents"])
