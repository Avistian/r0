"""Independent source-law formulas, raw custody and fit-parameter binding."""

import importlib
import json
from dataclasses import asdict, replace
from typing import Any

import numpy as np
import pytest

from r0.modules import priors as p


def pinned(family: str = "rf") -> p.FittedTreeParameterPrior:
    return p.FittedTreeParameterPrior(
        p.FittedTreePlan(family=family, input_width=2, rows=32),
        profile="tabicl-pinned-depth-count-v1",
    )


def test_pinned_full_support_tail_mass_and_raw_numpy_expression():
    rng = np.random.default_rng(141)
    expected = np.random.default_rng(141).exponential(2.0, size=(2000, 2))
    expected = np.minimum(4, np.floor(expected) + [2, 1]).astype(int)
    actual = np.array(
        [
            (s.plan.depth, s.plan.estimators)
            for s in (p.sample_fitted_tree_plan(pinned(), rng=rng) for _ in range(2000))
        ]
    )
    np.testing.assert_array_equal(actual, expected)
    assert set(actual[:, 0]) == {2, 3, 4}
    assert set(actual[:, 1]) == {1, 2, 3, 4}
    for column, offset in ((0, 2), (1, 1)):
        for value in range(offset, 5):
            # The last bin contains the entire exponential tail, not just [k,k+1).
            probability = np.exp(-0.5 * (value - offset))
            if value < 4:
                probability *= 1 - np.exp(-0.5)
            assert (
                abs(np.count_nonzero(actual[:, column] == value) / len(actual) - float(probability))
                < 0.04
            )


@pytest.mark.parametrize(
    "engine",
    [np.random.PCG64, np.random.PCG64DXSM, np.random.MT19937, np.random.Philox, np.random.SFC64],
)
def test_five_engines_record_reopen_and_custody_tamper(engine: Any):
    sample = p.sample_fitted_tree_plan(pinned("dt"), rng=np.random.Generator(engine(29)))
    assert sample.unused_parameters == ("estimators",)
    assert len(sample.raw) == 2
    assert sample.backend_semantics == "sklearn-independent-output-r0-v1"
    raw = json.loads(json.dumps(asdict(sample)))
    assert p.sampled_fitted_plan_from_dict(raw) == sample
    with pytest.raises(ValueError, match="limit|cap"):
        p.sampled_fitted_plan_from_dict(raw, limits=p.CausalLimits(parameters=1))
    for mutate in ("plan", "raw", "source", "lifecycle"):
        bad = json.loads(json.dumps(raw))
        if mutate == "plan":
            bad["plan"]["depth"] = 2 if sample.plan.depth != 2 else 3
        elif mutate == "raw":
            bad["raw"][0]["role"] = "changed"
        else:
            bad[mutate] = "claimed-source-lifecycle"
        with pytest.raises(ValueError, match="differ|provenance"):
            p.sampled_fitted_plan_from_dict(bad)


def test_extension_laws_and_incompatible_controls_before_draws():
    prior = p.FittedTreeParameterPrior(
        p.FittedTreePlan(family="gbdt", task="classification", rows=32),
        (
            ("depth", p.FittedParameterLaw(kind="capped-exponential", offset=1, cap=6, rate=0.7)),
            ("estimators", p.FittedParameterLaw(kind="integer-uniform", low=1, high=8)),
            (
                "classes",
                p.FittedParameterLaw(
                    kind="categorical", values=(2, 3, 5), probabilities=(0.2, 0.3, 0.5)
                ),
            ),
            ("rows", p.FittedParameterLaw(kind="fixed", values=(64,))),
            ("learning_rate", p.FittedParameterLaw(kind="log-uniform", low=0.03, high=0.8)),
        ),
    )
    sample = p.sample_fitted_tree_plan(prior, rng=np.random.default_rng(4))
    assert sample.plan.rows == 64
    assert 0.03 <= sample.plan.learning_rate <= 0.8
    assert sample.source == "r0-explicit-parameter-laws-v1"
    assert p.sampled_fitted_plan_from_dict(json.loads(json.dumps(asdict(sample)))) == sample
    for name in ("learning_rate", "classes", "estimators"):
        with pytest.raises(ValueError, match="unused"):
            p.FittedTreeParameterPrior(
                laws=((name, p.FittedParameterLaw(kind="fixed", values=(1,))),)
            )
    with pytest.raises(ValueError, match="unused"):
        p.FittedTreeParameterPrior(
            p.FittedTreePlan(family="rf"), (("splitter", p.FittedParameterLaw(values=("best",))),)
        )
    with pytest.raises(ValueError, match="finite"):
        p.FittedParameterLaw(kind="fixed", low=float("nan"))
    for rate in (0, -1, float("nan"), float("inf"), 1e-320):
        with pytest.raises(ValueError, match="exponential"):
            p.FittedParameterLaw(kind="capped-exponential", rate=rate)
    with pytest.raises(ValueError, match="overridden"):
        replace(pinned(), laws=(("depth", p.FittedParameterLaw(values=(3,))),))


def test_preallocation_limits_and_overflow_roll_back_parameter_stream():
    rng = np.random.default_rng(3)
    state = json.dumps(rng.bit_generator.state, sort_keys=True)
    with pytest.raises(ValueError, match="budget"):
        p.sample_fitted_tree_plan(replace(pinned(), raw_limits=p.RawDrawLimits(draws=1)), rng=rng)
    assert json.dumps(rng.bit_generator.state, sort_keys=True) == state
    prior = p.FittedTreeParameterPrior(
        laws=(
            (
                "depth",
                p.FittedParameterLaw(kind="capped-exponential", offset=1, cap=8, rate=1e-308),
            ),
        )
    )
    # This deterministic seed's first exponential exceeds the finite binary64 range.
    huge = np.random.default_rng(4)
    before = json.dumps(huge.bit_generator.state, sort_keys=True)
    with pytest.raises(ValueError, match="finite"):
        p.sample_fitted_tree_plan(prior, rng=huge)
    assert json.dumps(huge.bit_generator.state, sort_keys=True) == before


def skeleton() -> p.FunctionalCausalRecipe:
    return p.FunctionalCausalRecipe(
        p.LinearRecipe(rows=16, context_rows=8, features=2, signal_features=2, active_features=2),
        (
            p.FunctionalCausalNode(
                (), p.FrozenLinearFunction(((1.0, 0.0), (0.0, 1.0))), p.JointRootBase("gaussian", 2)
            ),
            p.FunctionalCausalNode((0,), p.FrozenLinearFunction(((1.0, -1.0),))),
        ),
        ((0, 0), (0, 1)),
        ((1, 0),),
    )


@pytest.mark.parametrize("family", ["dt", "rf", "extra_trees", "gbdt"])
@pytest.mark.parametrize("task", ["regression", "classification"])
def test_parameterized_fit_backend_plan_archive_and_selected_rows(family: str, task: str):
    prior = replace(
        pinned(family),
        base=p.FittedTreePlan(family=family, task=task, input_width=2, rows=32, classes=3),
    )
    streams = dict(
        zip(
            ("parameter_rng", "prefix_rng", "input_rng", "target_rng", "backend_rng"),
            (np.random.default_rng(i) for i in range(5)),
            strict=True,
        )
    )
    result = p.sample_parameterized_fitted_graph(skeleton(), ((1, prior),), **streams)
    assert result.parameter_samples[0][1].plan == result.fits[0].realization.plan
    actual = json.loads(result.fits[0].realization.backend_parameters[0])
    assert actual["max_depth"] == result.parameter_samples[0][1].plan.depth
    if family != "dt":
        assert actual["n_estimators"] == result.parameter_samples[0][1].plan.estimators
    generated = p.generate_task(result, seed=17, task_id=2**53 + 1)
    restored = p.recipe_from_dict(json.loads(generated.recipe_json)["recipe"])
    assert restored == result
    assert p.explain_generation(generated, [15, 0, 15])["rows"] == [15, 0, 15]
    changed = json.loads(generated.recipe_json)["recipe"]
    changed["parameter_samples"][0][1]["plan"]["depth"] = 1
    with pytest.raises(ValueError, match="custody"):
        p.recipe_from_dict(changed)


def test_fit_failure_rolls_back_all_five_streams(monkeypatch: pytest.MonkeyPatch):
    streams = {
        key: np.random.default_rng(i)
        for i, key in enumerate(
            ("parameter_rng", "prefix_rng", "input_rng", "target_rng", "backend_rng")
        )
    }
    before = {k: json.dumps(v.bit_generator.state, sort_keys=True) for k, v in streams.items()}

    def fail(*args: Any, **kwargs: Any) -> Any:
        raise RuntimeError("forced backend failure after parameter and population sampling")

    monkeypatch.setattr(
        importlib.import_module("sklearn.ensemble").RandomForestRegressor, "fit", fail
    )
    with pytest.raises(RuntimeError, match="forced backend"):
        p.sample_parameterized_fitted_graph(skeleton(), ((1, pinned()),), **streams)
    assert {
        k: json.dumps(v.bit_generator.state, sort_keys=True) for k, v in streams.items()
    } == before
