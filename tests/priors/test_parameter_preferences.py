"""Named sampling preferences retain scope, realizations and replayable laws."""

import json
from dataclasses import asdict

import numpy as np
import pytest

from r0.modules import priors


def test_global_numerical_preferences_reuse_values_and_reject_changed_contracts() -> None:
    parameter = priors.NumericalParameter
    sampler_type = priors.ParameterSampler
    decode = priors.parameter_preferences_from_dict
    sampler = sampler_type(np.random.default_rng(28), default_mode="global")
    request = parameter(name="gain", low=2.0, high=10.0, shape=(2,))
    first = sampler.numerical(request)
    second = sampler.numerical(request)
    assert first.values == second.values
    assert first.preference == second.preference
    assert len(first.values) == 2
    assert all(2 <= v < 10 for v in first.values)
    with pytest.raises(ValueError, match="global.*contract"):
        sampler.numerical(parameter(name="gain", low=0.0, high=1.0, shape=(2,)))
    record = sampler.freeze()
    assert len(record.draws) == 2
    assert asdict(decode(json.loads(json.dumps(asdict(record))))) == asdict(record)
    corrupted = json.loads(json.dumps(asdict(record)))
    corrupted["draws"][1]["values"][0] = 9.5
    with pytest.raises(ValueError, match="numerical.*realization|realization.*numerical"):
        decode(corrupted)


def test_meta_preferences_share_beta_law_while_local_draws_remain_independent() -> None:
    sampler = priors.ParameterSampler(np.random.default_rng(29), default_mode="meta")
    first = sampler.numerical(priors.NumericalParameter("gain", 0.0, 1.0, shape=(4,)))
    second = sampler.numerical(priors.NumericalParameter("gain", 2.0, 10.0, shape=(3,)))
    local = sampler.numerical(priors.NumericalParameter("gain", 0.0, 1.0, mode="local"))
    other = sampler.numerical(priors.NumericalParameter("gain", 0.0, 1.0, mode="local"))
    record = sampler.freeze()
    assert first.preference == second.preference
    assert local.preference == other.preference != first.preference
    assert local.units != other.units
    preference = record.preferences[first.preference]
    assert preference.mode == "meta"
    assert preference.mean is not None
    assert 0 < preference.mean < 1
    assert preference.concentration is not None
    assert 0.1 <= preference.concentration <= 10000
    assert preference.units == ()
    assert len(first.values) == 4
    assert len(second.values) == 3
    assert all(0 <= v <= 1 for v in first.values)
    assert all(2 <= v <= 10 for v in second.values)
    restored = priors.parameter_preferences_from_dict(json.loads(json.dumps(asdict(record))))
    assert restored == record
    corrupted = json.loads(json.dumps(asdict(record)))
    corrupted["draws"][2]["preference"] = first.preference
    with pytest.raises(ValueError, match="scope"):
        priors.parameter_preferences_from_dict(corrupted)
    replay = priors.ParameterSampler(np.random.default_rng(29), default_mode="meta")
    for draw in record.draws:
        assert replay.numerical(draw.request) == draw


def test_numerical_realizations_preserve_extreme_ranges_boundaries_and_integer_laws() -> None:
    parameter, draw = priors.NumericalParameter, priors.NumericalDraw
    assert draw(parameter("extreme", -1e308, 1e308), 0, (0.5,), (0.0,)).values == (0.0,)
    assert draw(parameter("log", 1e-300, 1e300, scale="log"), 0, (0.5,), (1.0,)).values == (1.0,)
    rounded = parameter("round", 0.0, 4.0, shape=(4,), integer="round")
    assert draw(rounded, 0, (0.125, 0.375, 0.625, 0.875), (0, 2, 2, 4)).values == (0, 2, 2, 4)
    floored = parameter("floor", 0, 5, shape=(3,), integer="floor")
    assert draw(floored, 0, (0.0, 0.5, 1.0), (0, 2, 4)).values == (0, 2, 4)
    boundary = parameter("atoms", 0.0, 1.0, shape=(5,), boundary_mass=True)
    assert draw(boundary, 0, (0.0, 0.05, 0.5, 0.95, 1.0), (0.0, 0.0, 0.5, 1.0, 1.0)).values == (
        0.0,
        0.0,
        0.5,
        1.0,
        1.0,
    )
    with pytest.raises(ValueError, match="positive"):
        parameter("invalid-log", 0.0, 1.0, scale="log")
    with pytest.raises(ValueError, match="integer bounds"):
        parameter("invalid-floor", 0.5, 5, integer="floor")
    with pytest.raises(ValueError, match="realization"):
        draw(rounded, 0, (0.125, 0.375, 0.625, 0.875), (1, 2, 3, 4))
    sampler = priors.ParameterSampler(np.random.default_rng(30), default_mode="local")
    sampler.numerical(parameter("sampled", 0.001, 10.0, scale="log", boundary_mass=True))
    record = sampler.freeze()
    assert priors.parameter_preferences_from_dict(json.loads(json.dumps(asdict(record)))) == record


def test_global_quantile_policy_remaps_bounds_but_rejects_changed_batch_shape() -> None:
    rng = np.random.default_rng(31)
    sampler = priors.ParameterSampler(rng, default_mode="global", global_policy="quantile")
    first = sampler.numerical(priors.NumericalParameter("gain", 0.0, 1.0))
    remapped = sampler.numerical(priors.NumericalParameter("gain", 10.0, 20.0))
    explicit = sampler.numerical(priors.NumericalParameter("gain", 10.0, 20.0, mode="global"))
    assert first.units == remapped.units == explicit.units
    assert first.preference == remapped.preference == explicit.preference
    assert 10 <= remapped.values[0] <= 20
    assert first.values != remapped.values
    before = json.dumps(rng.bit_generator.state, sort_keys=True)
    with pytest.raises(ValueError, match="global.*contract"):
        sampler.numerical(priors.NumericalParameter("gain", 10.0, 20.0, shape=(2,)))
    assert json.dumps(rng.bit_generator.state, sort_keys=True) == before
    record = sampler.freeze()
    assert len(record.draws) == 3
    assert priors.parameter_preferences_from_dict(json.loads(json.dumps(asdict(record)))) == record
    corrupted = json.loads(json.dumps(asdict(record)))
    corrupted["global_policy"] = "strict"
    with pytest.raises(ValueError, match="global.*contract"):
        priors.parameter_preferences_from_dict(corrupted)


def test_preference_limits_reject_before_random_draws_and_check_frozen_record_cost() -> None:
    limits = priors.PreferenceLimits(max_draws=2, max_values=3, max_parameters=30)
    rng = np.random.default_rng(32)
    sampler = priors.ParameterSampler(rng, default_mode="global", limits=limits)
    request = priors.NumericalParameter("gain", 0.0, 1.0, shape=(2,))
    sampler.numerical(request)
    record = sampler.freeze()
    # Preference: two bounds + one shape dimension + two units = 5.
    # Draw: two bounds + one shape dimension + index + two units + two values = 8.
    assert record.parameter_count == 13
    before = json.dumps(rng.bit_generator.state, sort_keys=True)
    with pytest.raises(ValueError, match="value limit"):
        sampler.numerical(priors.NumericalParameter("other", 0.0, 1.0, shape=(2,)))
    assert json.dumps(rng.bit_generator.state, sort_keys=True) == before
    assert sampler.freeze() == record
    sampler.numerical(priors.NumericalParameter("last", 0.0, 1.0))
    before = json.dumps(rng.bit_generator.state, sort_keys=True)
    with pytest.raises(ValueError, match="draw limit"):
        sampler.numerical(priors.NumericalParameter("last", 0.0, 1.0))
    assert json.dumps(rng.bit_generator.state, sort_keys=True) == before
    encoded = json.loads(json.dumps(asdict(record)))
    assert priors.parameter_preferences_from_dict(encoded) == record
    encoded["limits"]["max_parameters"] = 12
    with pytest.raises(ValueError, match="parameter limit"):
        priors.parameter_preferences_from_dict(encoded)
    small = priors.ParameterSampler(rng, limits=priors.PreferenceLimits(max_parameters=1))
    with pytest.raises(ValueError, match="parameter limit"):
        small.numerical(priors.NumericalParameter("big", 0.0, 1.0, shape=(100000,)))
    assert json.dumps(rng.bit_generator.state, sort_keys=True) == before
    encoded = json.loads(json.dumps(asdict(record)))
    encoded["preferences"][0]["request"]["high"] = 2.0
    with pytest.raises(ValueError, match="first request|contract"):
        priors.parameter_preferences_from_dict(encoded)
