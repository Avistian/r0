"""Population isolation and retained fit semantics through public entry points."""

import copy
from dataclasses import asdict, replace
from typing import Any

import numpy as np
import pytest
import torch
from numpy.typing import NDArray

from r0.modules import priors as p


def skeleton(rows: int = 32, support: int = 8, ball: bool = False):
    task = p.LinearRecipe(
        rows=rows, features=2, signal_features=2, active_features=2, context_rows=support
    )
    root = p.FunctionalCausalNode(
        (),
        p.FrozenLinearFunction(((1.0, 0.0), (0.0, 1.0))),
        root=p.JointRootBase("ball" if ball else "gaussian", 2),
    )
    leaf = p.FunctionalCausalNode((0,), p.FrozenLinearFunction(((1.0, 1.0),)))
    return p.FunctionalCausalRecipe(task, (root, leaf), ((0, 0), (0, 1)), ((1, 0),))


def fit_graph(
    recipe: p.FunctionalCausalRecipe,
    mode: str = "support-only",
    family: str = "dt",
    task: str = "regression",
    mixed: p.MixedFitPopulation | None = None,
):
    rows = recipe.task.context_rows if mode == "support-only" else recipe.task.rows
    plan = p.PopulationTreePlan(family=family, task=task, rows=rows, input_width=2)
    return p.fit_population_graph(
        recipe,
        ((1, plan),),
        population=mode,
        seed=29,
        task_id=71,
        mixed=mixed,
        input_rng=np.random.default_rng(1),
        target_rng=np.random.default_rng(2),
        backend_rng=np.random.default_rng(3),
    )


@pytest.mark.parametrize("family", ["dt", "rf", "extra_trees", "gbdt"])
@pytest.mark.parametrize("task", ["regression", "classification"])
def test_support_exact_original_rows_and_archive(family: str, task: str):
    graph = fit_graph(skeleton(), family=family, task=task)
    generated = p.generate_task(graph, seed=29, task_id=71)
    fit = graph.fits[0].realization
    assert np.array_equal(generated.inputs[generated.context_indices].numpy(), fit.inputs)
    assert p.recipe_from_dict(asdict(graph)) == graph
    assert p.population_fit_from_dict(asdict(fit)) == fit
    before = p.generate_task(graph, seed=29, task_id=71)
    after = p.generate_task(p.recipe_from_dict(asdict(graph)), seed=29, task_id=71)
    assert torch.equal(before.targets, after.targets)
    bad = asdict(graph)
    bad["fits"][0]["realization"]["inputs"] = ((999.0, 999.0),) * fit.plan.rows
    with pytest.raises(ValueError, match="original episode"):
        p.recipe_from_dict(bad)


@pytest.mark.parametrize("ball", [False, True])
def test_query_append_does_not_change_support_fit(ball: bool):
    small = fit_graph(skeleton(rows=32, ball=ball))
    large = fit_graph(skeleton(rows=61, ball=ball))
    assert small.fits == large.fits
    a, b = (p.generate_task(g, seed=29, task_id=71) for g in (small, large))
    assert torch.equal(a.inputs[a.context_indices], b.inputs[b.context_indices])
    assert torch.equal(a.scores[a.context_indices], b.scores[b.context_indices])


def test_batch_reference_has_new_fit_identity():
    a = fit_graph(skeleton(), mode="batch-refit-reference")
    b = fit_graph(skeleton(rows=48), mode="batch-refit-reference")
    assert a.fits[0].realization.inputs != b.fits[0].realization.inputs
    assert a.fits[0].realization.function != b.fits[0].realization.function
    emitted = p.generate_task(a, seed=29, task_id=71)
    assert np.array_equal(emitted.inputs.numpy(), a.fits[0].realization.inputs)


@pytest.mark.parametrize("family", ["dt", "rf", "extra_trees", "gbdt"])
@pytest.mark.parametrize("task", ["regression", "classification"])
def test_mixed_population_is_separate_replayable_and_native_categorical(family: str, task: str):
    mixed = p.MixedFitPopulation(categorical_fraction=(0.5, 0.5), probabilities="dirichlet-r0")
    graph = fit_graph(
        skeleton(rows=48), mode="mixed-independent", family=family, task=task, mixed=mixed
    )
    fit = graph.fits[0].realization
    assert sum(c > 0 for c in fit.cardinalities) == 1
    for column, cardinality in enumerate(fit.cardinalities):
        if cardinality:
            assert all(
                v[column] == int(v[column]) and 0 <= v[column] < cardinality for v in fit.inputs
            )
    assert p.recipe_from_dict(asdict(graph)) == graph
    bad = asdict(fit)
    bad["mixed"]["gaussian_mean"] = 19.0
    with pytest.raises(ValueError, match="custody"):
        p.population_fit_from_dict(bad)
    if task == "classification":
        assert set(row[0] for row in fit.targets) <= {0.0, 1.0}


def test_one_row_supported_and_boost_single_class_rolls_back():
    graph = fit_graph(skeleton(support=1))
    assert graph.fits[0].realization.plan.rows == 1
    streams = [np.random.default_rng(i) for i in range(3)]
    before = [copy.deepcopy(r.bit_generator.state) for r in streams]
    with pytest.raises(ValueError, match="two observed classes"):
        p.fit_population_tree(
            p.PopulationTreePlan(rows=1, family="gbdt", task="classification"),
            population="support-only",
            inputs=np.ones((1, 1)),
            input_rng=streams[0],
            target_rng=streams[1],
            backend_rng=streams[2],
        )
    assert before == [r.bit_generator.state for r in streams]


def test_old_plan_keeps_its_minimum_and_mixed_cap_never_clips():
    with pytest.raises(ValueError, match="envelope"):
        p.FittedTreePlan(rows=1)
    streams = [np.random.default_rng(i) for i in range(3)]
    before = [copy.deepcopy(r.bit_generator.state) for r in streams]
    with pytest.raises(ValueError, match="exceeds admitted cap"):
        p.fit_population_tree(
            p.PopulationTreePlan(rows=32),
            population="mixed-independent",
            mixed=p.MixedFitPopulation(
                categorical_fraction=(0.5, 0.5), geometric_probability=1e-10
            ),
            input_rng=streams[0],
            target_rng=streams[1],
            backend_rng=streams[2],
        )
    assert before == [r.bit_generator.state for r in streams]


def test_caller_limits_reject_before_fitting_and_decoding():
    graph = fit_graph(skeleton())
    with pytest.raises(ValueError, match="limit"):
        p.population_fit_from_dict(
            asdict(graph.fits[0].realization), limits=replace(p.CausalLimits(), parameters=20)
        )


def test_mixed_raw_population_matches_independent_numpy_oracle():
    import math

    mixed = p.MixedFitPopulation(
        categorical_fraction=(0.5, 0.5),
        gaussian_mean=2.0,
        gaussian_scale=3.0,
        probabilities="dirichlet-r0",
        concentration=0.7,
    )
    plan = p.PopulationTreePlan(rows=32, input_width=3)
    fit = p.fit_population_tree(
        plan,
        population="mixed-independent",
        mixed=mixed,
        input_rng=np.random.default_rng(77),
        target_rng=np.random.default_rng(9),
        backend_rng=np.random.default_rng(4),
    )
    rng = np.random.default_rng(77)
    selected = set(rng.permutation(3)[:2])
    columns: list[NDArray[Any]] = []
    for column in range(3):
        if column not in selected:
            columns.append(2 + 3 * rng.normal(size=32))
        else:
            cardinality = 1 + math.floor(math.log1p(-rng.random()) / math.log(0.5))
            weights = rng.gamma(0.7, size=cardinality)
            columns.append(rng.choice(cardinality, size=32, p=weights / weights.sum()))
    assert np.array_equal(np.column_stack(columns), fit.inputs)
    assert np.array_equal(np.random.default_rng(9).normal(size=(32, 1)), fit.targets)
    assert rng.bit_generator.state == fit.final_rng[0].restore().bit_generator.state


def test_support_intervention_preserves_original_population_and_frozen_fit():
    graph = fit_graph(skeleton(ball=True))
    pair = p.generate_causal_pair(
        graph, interventions=(p.HardIntervention(1, (0,), (7.0,)),), seed=29, task_id=71
    )
    assert torch.equal(pair.before.inputs, pair.after.inputs)
    assert torch.all(pair.after.scores == 7)


def test_failed_downstream_fit_rolls_back_all_callers(monkeypatch: pytest.MonkeyPatch):
    streams = [np.random.default_rng(i) for i in range(3)]
    before = [copy.deepcopy(r.bit_generator.state) for r in streams]

    def fail(*args: object, **kwargs: object):
        raise RuntimeError("forced backend failure")

    monkeypatch.setattr("r0.modules.priors.lib.fitted_population_fit.fit_backend", fail)
    with pytest.raises(RuntimeError, match="forced backend"):
        p.fit_population_graph(
            skeleton(),
            ((1, p.PopulationTreePlan(rows=8, input_width=2)),),
            population="support-only",
            seed=29,
            task_id=71,
            input_rng=streams[0],
            target_rng=streams[1],
            backend_rng=streams[2],
        )
    assert before == [r.bit_generator.state for r in streams]


@pytest.mark.parametrize(
    "engine",
    [np.random.PCG64, np.random.PCG64DXSM, np.random.MT19937, np.random.Philox, np.random.SFC64],
)
def test_population_raw_replay_all_engines(engine: type[np.random.BitGenerator]):
    fit = p.fit_population_tree(
        p.PopulationTreePlan(
            rows=32, input_width=2, output_width=3, task="classification", classes=100
        ),
        population="mixed-independent",
        mixed=p.MixedFitPopulation(categorical_fraction=(0.5, 0.5), target_cardinality="geometric"),
        input_rng=np.random.Generator(engine(77)),
        target_rng=np.random.Generator(engine(1)),
        backend_rng=np.random.Generator(engine(4)),
    )
    assert p.population_fit_from_dict(asdict(fit)) == fit
    assert len(fit.function.outputs) == 3


def test_archive_does_not_refit_and_rejects_source_label_tamper(monkeypatch: pytest.MonkeyPatch):
    graph = fit_graph(skeleton())

    def fail(*args: object, **kwargs: object):
        raise AssertionError("archive must not fit")

    monkeypatch.setattr("r0.modules.priors.lib.fitted_population_fit.fit_backend", fail)
    assert p.recipe_from_dict(asdict(graph)) == graph
    bad = asdict(graph)
    bad["fits"][0]["realization"]["source_contract"] = "full-paper-reproduction"
    with pytest.raises(ValueError, match="provenance"):
        p.recipe_from_dict(bad)
