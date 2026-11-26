"""Named categorical laws retain choices, draws and sampling scope."""

import json
from dataclasses import asdict

import numpy as np
import pytest

from r0.modules import priors


def test_categorical_global_reuse_and_local_draws_keep_replayable_choice_contracts() -> None:
    request = priors.CategoricalParameter("matrix", ("gaussian", "kernel", "factorized"))
    rng = np.random.default_rng(2801)
    sampler = priors.CategoricalSampler(rng, default_mode="global")
    first = sampler.categorical(request)
    before = json.dumps(rng.bit_generator.state, sort_keys=True)
    second = sampler.categorical(request)
    assert first == second
    assert first.value == request.choices[first.index]
    assert json.dumps(rng.bit_generator.state, sort_keys=True) == before
    with pytest.raises(ValueError, match="categorical.*contract"):
        sampler.categorical(priors.CategoricalParameter("matrix", ("kernel", "gaussian")))
    assert json.dumps(rng.bit_generator.state, sort_keys=True) == before
    record = sampler.freeze()
    assert record.parameter_count == 7
    assert len(record.draws) == 2
    assert (
        priors.categorical_preferences_from_dict(json.loads(json.dumps(asdict(record)))) == record
    )
    corrupted = json.loads(json.dumps(asdict(record)))
    corrupted["draws"][1]["index"] = (first.index + 1) % 3
    corrupted["draws"][1]["value"] = request.choices[(first.index + 1) % 3]
    with pytest.raises(ValueError, match="categorical.*realization"):
        priors.categorical_preferences_from_dict(corrupted)
    local = priors.CategoricalParameter("matrix", request.choices, mode="local")
    a, b = sampler.categorical(local), sampler.categorical(local)
    assert a.preference == b.preference != first.preference
    assert a.unit != b.unit
    replay = priors.CategoricalSampler(np.random.default_rng(2801), default_mode="global")
    for draw in sampler.freeze().draws:
        assert replay.categorical(draw.request) == draw
