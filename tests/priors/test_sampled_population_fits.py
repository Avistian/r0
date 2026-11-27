"""Sampled parameter laws bind actual lifecycle plans and preserve four-stream custody."""

import copy
from dataclasses import asdict, replace

import numpy as np
import pytest

from r0.modules import priors as p


def skeleton(rows: int = 32, context: int = 16) -> p.FunctionalCausalRecipe:
    return p.FunctionalCausalRecipe(
        p.LinearRecipe(
            rows=rows, context_rows=context, features=1, signal_features=1, active_features=1
        ),
        (
            p.FunctionalCausalNode(
                (), p.FrozenLinearFunction(((1.0,),)), p.JointRootBase("ball", 1)
            ),
            p.FunctionalCausalNode((0,), p.FrozenLinearFunction(((1.0,),))),
        ),
        ((0, 0),),
        ((1, 0),),
    )


def streams() -> list[np.random.Generator]:
    return [np.random.default_rng(seed) for seed in (17, 18, 19, 20)]


def construct(
    recipe: p.FunctionalCausalRecipe,
    prior: p.PopulationTreeParameterPrior,
    rngs: list[np.random.Generator],
) -> p.SampledPopulationFittedGraph:
    return p.sample_parameterized_population_graph(
        recipe,
        ((1, prior),),
        population=prior.population,
        seed=7,
        task_id=9,
        parameter_rng=rngs[0],
        input_rng=rngs[1],
        target_rng=rngs[2],
        backend_rng=rngs[3],
        mixed=p.MixedFitPopulation() if prior.population == "mixed-independent" else None,
    )


@pytest.mark.parametrize(
    "population", ["support-only", "batch-refit-reference", "mixed-independent"]
)
@pytest.mark.parametrize("family", ["dt", "extra_trees", "rf", "gbdt"])
@pytest.mark.parametrize("task", ["regression", "classification"])
def test_every_backend_task_lifecycle_pinned_binding(population: str, family: str, task: str):
    rows = 16 if population == "support-only" else 32
    prior = p.PopulationTreeParameterPrior(
        base=p.PopulationTreePlan(rows=rows, family=family, task=task),
        population=population,
        profile="tabicl-pinned-depth-count-v1",
    )
    graph = construct(skeleton(), prior, streams())
    sample = graph.parameter_samples[0][1]
    assert sample.plan == graph.fits[0].realization.plan
    assert sample.unused_parameters == (("estimators",) if family == "dt" else ())
    oracle = np.random.default_rng(17)
    assert sample.plan.depth == min(4, 2 + int(np.floor(oracle.exponential(2))))
    assert sample.plan.estimators == min(4, 1 + int(np.floor(oracle.exponential(2))))
    assert oracle.bit_generator.state == sample.final_rng.restore().bit_generator.state
    assert sample.plan.rows == rows
    assert p.recipe_from_dict(asdict(graph)) == graph
    assert p.sampled_population_plan_from_dict(asdict(sample)) == sample
    changed = asdict(graph)
    changed["parameter_samples"][0][1]["plan"]["depth"] = 8
    with pytest.raises(ValueError, match="custody"):
        p.recipe_from_dict(changed)


def test_support_query_append_keeps_parameter_raw_and_actual_fit_unchanged():
    prior = p.PopulationTreeParameterPrior(
        base=p.PopulationTreePlan(rows=1), profile="tabicl-pinned-depth-count-v1"
    )
    a = construct(skeleton(32, 1), prior, streams())
    b = construct(skeleton(93, 1), prior, streams())
    assert a.parameter_samples == b.parameter_samples
    assert a.fits == b.fits
    assert a.fits[0].realization.plan.rows == 1


@pytest.mark.parametrize("population", ["support-only", "batch-refit-reference"])
def test_unused_row_laws_and_wrong_actual_rows_reject_before_draws(population: str):
    rngs = streams()
    before = [copy.deepcopy(r.bit_generator.state) for r in rngs]
    with pytest.raises(ValueError, match="rows law is unused"):
        p.PopulationTreeParameterPrior(
            population=population, laws=(("rows", p.FittedParameterLaw(values=(32,))),)
        )
    prior = p.PopulationTreeParameterPrior(
        base=p.PopulationTreePlan(rows=33), population=population
    )
    with pytest.raises(ValueError, match="actual episode rows"):
        construct(skeleton(), prior, rngs)
    assert before == [r.bit_generator.state for r in rngs]


@pytest.mark.parametrize("kind", ["fixed", "categorical", "integer-uniform", "capped-exponential"])
def test_mixed_rows_can_be_sampled_down_to_one(kind: str):
    law = p.FittedParameterLaw(kind=kind, values=(1,), low=1, high=1, offset=1, cap=1)
    prior = p.PopulationTreeParameterPrior(population="mixed-independent", laws=(("rows", law),))
    graph = construct(skeleton(), prior, streams())
    assert graph.parameter_samples[0][1].plan.rows == 1
    assert len(graph.fits[0].realization.inputs) == 1


def test_failure_after_parameter_draws_rolls_back_every_stream(monkeypatch: pytest.MonkeyPatch):
    prior = p.PopulationTreeParameterPrior(
        base=p.PopulationTreePlan(rows=16), profile="tabicl-pinned-depth-count-v1"
    )
    rngs = streams()
    before = [copy.deepcopy(r.bit_generator.state) for r in rngs]

    def fail(*args: object, **kwargs: object):
        raise RuntimeError("forced backend failure after parameter selection")

    monkeypatch.setattr("r0.modules.priors.lib.fitted_population_fit.fit_backend", fail)
    with pytest.raises(RuntimeError, match="forced backend"):
        construct(skeleton(), prior, rngs)
    assert before == [r.bit_generator.state for r in rngs]


def test_worst_case_admission_and_unused_parameter_choices_do_not_consume_rng():
    prior = p.PopulationTreeParameterPrior(
        base=p.PopulationTreePlan(rows=16), profile="tabicl-pinned-depth-count-v1"
    )
    rngs = streams()
    before = [copy.deepcopy(r.bit_generator.state) for r in rngs]
    recipe = replace(skeleton(), limits=replace(p.CausalLimits(), working_bytes=2**20))
    with pytest.raises(ValueError, match="metadata|memory"):
        construct(recipe, prior, rngs)
    assert before == [r.bit_generator.state for r in rngs]
    with pytest.raises(ValueError, match="unused"):
        p.PopulationTreeParameterPrior(
            laws=(("learning_rate", p.FittedParameterLaw(kind="uniform", low=0.1, high=0.8)),)
        )


def test_population_and_row_binding_tamper_rejected_without_backend(
    monkeypatch: pytest.MonkeyPatch,
):
    graph = construct(
        skeleton(), p.PopulationTreeParameterPrior(base=p.PopulationTreePlan(rows=16)), streams()
    )

    def fail(*args: object, **kwargs: object):
        raise AssertionError("no backend during archive replay")

    monkeypatch.setattr("r0.modules.priors.lib.fitted_population_fit.fit_backend", fail)
    assert p.recipe_from_dict(asdict(graph)) == graph
    bad = asdict(graph.parameter_samples[0][1])
    bad["row_binding"] = "independent-selected-fit-rows-v1"
    with pytest.raises(ValueError, match="effective rows"):
        p.sampled_population_plan_from_dict(bad)


@pytest.mark.parametrize(
    "engine",
    [np.random.PCG64, np.random.PCG64DXSM, np.random.MT19937, np.random.Philox, np.random.SFC64],
)
def test_parameter_raw_custody_all_engines(engine: type[np.random.BitGenerator]):
    prior = p.PopulationTreeParameterPrior(
        base=p.PopulationTreePlan(rows=16, family="gbdt"),
        laws=(("learning_rate", p.FittedParameterLaw(kind="log-uniform", low=0.01, high=0.7)),),
    )
    sample = p.sample_population_tree_plan(prior, rng=np.random.Generator(engine(7)))
    assert p.sampled_population_plan_from_dict(asdict(sample)) == sample
