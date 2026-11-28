"""Source capped-exponential laws checked against an independent NumPy oracle."""

import json
from dataclasses import asdict, replace
from typing import Any

import numpy as np
import pytest

from r0.modules import priors as p


@pytest.mark.parametrize(
    "engine",
    [np.random.PCG64, np.random.PCG64DXSM, np.random.MT19937, np.random.Philox, np.random.SFC64],
)
def test_xgboost_source_laws_replay_and_exact_stream(engine: Any) -> None:
    prior = p.XGBoostParameterPrior(p.XGBoostFitPlan(input_width=2, output_width=2, rows=32))
    rng = np.random.Generator(engine(29))
    oracle = np.random.Generator(engine(29))
    for _ in range(12):
        expected = np.minimum(4, [2, 1] + np.floor(oracle.exponential(2, size=2))).astype(int)
        sample = p.sample_xgboost_plan(prior, rng=rng)
        assert (sample.plan.depth, sample.plan.estimators) == tuple(expected)
        assert len(sample.raw) == 2
        assert p.sampled_xgboost_plan_from_dict(json.loads(json.dumps(asdict(sample)))) == sample

    def serialize(state: Any) -> str:
        return json.dumps(state, default=lambda value: value.tolist())

    assert serialize(rng.bit_generator.state) == serialize(oracle.bit_generator.state)


def test_xgboost_prior_rejects_before_drawing_and_detects_tampering():
    prior = p.XGBoostParameterPrior(p.XGBoostFitPlan(input_width=2, output_width=2, rows=32))
    rng = np.random.default_rng(41)
    before = json.dumps(rng.bit_generator.state)
    with pytest.raises(ValueError, match="limit|cap"):
        p.sample_xgboost_plan(prior, rng=rng, limits=p.CausalLimits(parameters=1))
    assert json.dumps(rng.bit_generator.state) == before
    sample = p.sample_xgboost_plan(prior, rng=rng)
    with pytest.raises(ValueError, match="replay"):
        replace(
            sample, plan=replace(sample.plan, estimators=1 if sample.plan.estimators != 1 else 2)
        )
    with pytest.raises(ValueError, match="source"):
        replace(sample, source="claimed-unrelated-paper")


def test_xgboost_extension_rates_and_caps_preserve_raw_tail_mass():
    prior = p.XGBoostParameterPrior(
        p.XGBoostFitPlan(input_width=2, output_width=2, rows=32),
        depth_rate=0.01,
        count_rate=0.02,
        depth_cap=6,
        count_cap=8,
        profile="r0-capped-exponential-xgboost-v1",
    )
    sample = p.sample_xgboost_plan(prior, rng=np.random.default_rng(7))
    assert (sample.plan.depth, sample.plan.estimators) == (6, 8)
    assert sample.raw[0].values[0] > 4
    assert sample.raw[1].values[0] > 7
    with pytest.raises(ValueError, match="pinned"):
        replace(prior, profile="tabicl-pinned-depth-count-v1")


def test_xgboost_parameter_decoder_pre_admits_before_snapshot_restore(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from r0.modules.priors.lib.generator_snapshot import GeneratorSnapshot

    sample = p.sample_xgboost_plan(
        p.XGBoostParameterPrior(p.XGBoostFitPlan(input_width=2, output_width=2, rows=32)),
        rng=np.random.default_rng(3),
    )

    def forbidden(*args: object, **kwargs: object) -> None:
        raise AssertionError("restored before caller admission")

    monkeypatch.setattr(GeneratorSnapshot, "restore", forbidden)
    with pytest.raises(ValueError, match="limit|cap"):
        p.sampled_xgboost_plan_from_dict(asdict(sample), limits=p.CausalLimits(parameters=1))


def test_sampled_parameters_drive_actual_xgboost_fit_and_reopen():
    sample = p.sample_xgboost_plan(
        p.XGBoostParameterPrior(p.XGBoostFitPlan(input_width=2, output_width=2, rows=32)),
        rng=np.random.default_rng(11),
    )
    fit = p.fit_xgboost(
        sample.plan,
        input_rng=np.random.default_rng(12),
        target_rng=np.random.default_rng(13),
        backend_rng=np.random.default_rng(14),
    )
    assert fit.plan == sample.plan
    assert json.loads(fit.backend_parameters)["max_depth"] == sample.plan.depth
    assert p.xgboost_fit_from_dict(json.loads(json.dumps(asdict(fit)))) == fit
