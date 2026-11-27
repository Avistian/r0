"""Independent synthetic fitting, raw custody and rollback against actual CPU backend."""

import copy
import importlib
from dataclasses import asdict, replace
from typing import Any

import numpy as np
import pytest

from r0.modules.priors import (
    CausalLimits,
    RawDrawLimits,
    XGBoostFitPlan,
    evaluate_xgboost_vector,
    fit_xgboost,
)


@pytest.mark.parametrize("objective", ["reg:squarederror", "binary:logistic", "multi:softprob"])
@pytest.mark.parametrize("engine", ["PCG64", "PCG64DXSM", "Philox", "SFC64", "MT19937"])
def test_frozen_synthetic_fit_matches_independent_raw_refit(objective: str, engine: str) -> None:
    constructor = getattr(np.random, engine)
    plan = XGBoostFitPlan(rows=32, estimators=2, depth=2, objective=objective, output_width=3)
    streams = [np.random.Generator(constructor(seed)) for seed in (4, 5, 6)]
    fit = fit_xgboost(plan, input_rng=streams[0], target_rng=streams[1], backend_rng=streams[2])
    oracle = [np.random.Generator(constructor(seed)) for seed in (4, 5, 6)]
    x = oracle[0].standard_normal((32, 3))
    y = (
        oracle[1].standard_normal((32, 3))
        if objective == "reg:squarederror"
        else oracle[1].integers(
            0, 3 if objective == "multi:softprob" else 2, size=(32, plan.target_width)
        )
    )
    seed = int(oracle[2].integers(0, 2**31, size=1)[0])
    np.testing.assert_array_equal(fit.raw[0].values, x.ravel())
    np.testing.assert_array_equal(fit.raw[1].values, y.ravel())
    assert fit.raw[2].values == (seed,)
    for left, right in zip(streams, oracle, strict=True):
        np.testing.assert_equal(left.bit_generator.state, right.bit_generator.state)
    backend = importlib.import_module("xgboost")
    refit = backend.train(
        plan.backend_parameters(seed), backend.DMatrix(x, label=y), num_boost_round=2
    )
    query = np.random.default_rng(7).normal(size=(13, 3))
    np.testing.assert_allclose(
        evaluate_xgboost_vector(fit.function, query),
        refit.predict(backend.DMatrix(query)),
        rtol=3e-6,
        atol=3e-7,
    )
    assert bytes(refit.save_raw(raw_format="json")).decode("ascii") == fit.model_json
    from r0.modules.priors import xgboost_fit_from_dict

    assert xgboost_fit_from_dict(asdict(fit)) == fit
    for key in asdict(fit):
        broken = asdict(fit)
        del broken[key]
        with pytest.raises(ValueError, match="fields"):
            xgboost_fit_from_dict(broken)
    with pytest.raises(ValueError, match="controls"):
        replace(fit, backend_parameters="{}")


def test_fit_backend_failure_rolls_back_all_three_streams(monkeypatch: pytest.MonkeyPatch) -> None:
    backend = importlib.import_module("xgboost")
    streams = [np.random.default_rng(seed) for seed in (1, 2, 3)]
    before = [copy.deepcopy(s.bit_generator.state) for s in streams]

    def fail(*args: object, **kwargs: object) -> None:
        raise RuntimeError("injected backend failure")

    monkeypatch.setattr(backend, "train", fail)
    with pytest.raises(RuntimeError, match="injected"):
        fit_xgboost(
            XGBoostFitPlan(rows=32),
            input_rng=streams[0],
            target_rng=streams[1],
            backend_rng=streams[2],
        )
    assert [s.bit_generator.state for s in streams] == before


@pytest.mark.parametrize(
    "raw_limits",
    [RawDrawLimits(numbers=160), RawDrawLimits(draws=2), RawDrawLimits(working_bytes=1024)],
)
def test_fit_raw_rejection_precedes_backend_import(
    raw_limits: RawDrawLimits, monkeypatch: pytest.MonkeyPatch
) -> None:
    def fail(name: str) -> None:
        raise AssertionError("backend import reached before admission")

    monkeypatch.setattr(importlib, "import_module", fail)
    with pytest.raises(ValueError, match="raw fit limit"):
        fit_xgboost(
            XGBoostFitPlan(rows=32),
            input_rng=np.random.default_rng(1),
            target_rng=np.random.default_rng(2),
            backend_rng=np.random.default_rng(3),
            raw_limits=raw_limits,
        )


def test_histogram_working_limit_rejects_before_backend() -> None:
    with pytest.raises(ValueError, match="histogram|working|parameter"):
        XGBoostFitPlan(input_width=256, output_width=100, depth=8).admit(
            CausalLimits(), RawDrawLimits()
        )


@pytest.mark.parametrize("objective", ["reg:squarederror", "binary:logistic"])
def test_one_output_hist_fit_uses_explicit_degenerate_vector(objective: str) -> None:
    plan = XGBoostFitPlan(output_width=1, objective=objective, rows=32, estimators=2)
    fit = fit_xgboost(
        plan,
        input_rng=np.random.default_rng(31),
        target_rng=np.random.default_rng(32),
        backend_rng=np.random.default_rng(33),
    )
    backend = importlib.import_module("xgboost")
    native = backend.Booster()
    native.load_model(bytearray(fit.model_json.encode("ascii")))
    query = np.random.default_rng(34).normal(size=(11, 3))
    expected = native.predict(backend.DMatrix(query)).reshape(-1, 1)
    np.testing.assert_allclose(
        evaluate_xgboost_vector(fit.function, query), expected, rtol=3e-6, atol=3e-7
    )


def test_fit_archive_rejects_every_missing_nested_record_field() -> None:
    from r0.modules.priors import xgboost_fit_from_dict

    fit = fit_xgboost(
        XGBoostFitPlan(rows=32, estimators=1),
        input_rng=np.random.default_rng(1),
        target_rng=np.random.default_rng(2),
        backend_rng=np.random.default_rng(3),
    )
    original = asdict(fit)
    paths = [
        ("plan",),
        ("limits",),
        ("raw_limits",),
        ("function",),
        ("initial_rng", 0),
        ("final_rng", 0),
        ("raw", 0),
        ("function", "trees", 0),
    ]
    for path in paths:
        mapping: Any = original
        for key in path:
            mapping = mapping[key]
        for field in mapping:
            altered = copy.deepcopy(original)
            target: Any = altered
            for key in path:
                target = target[key]
            del target[field]
            with pytest.raises(ValueError, match="fields"):
                xgboost_fit_from_dict(altered)
