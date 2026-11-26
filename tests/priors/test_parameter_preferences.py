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
