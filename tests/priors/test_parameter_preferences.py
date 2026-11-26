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
