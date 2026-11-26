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


def test_meta_categorical_preferences_retain_shared_or_refreshed_polynomial_laws() -> None:
    request = priors.CategoricalParameter("matrix", ("gaussian", "kernel", "factorized"))
    shared = priors.CategoricalSampler(np.random.default_rng(2802), default_mode="meta")
    first, second = shared.categorical(request), shared.categorical(request)
    record = shared.freeze()
    assert first.preference == second.preference
    assert first.unit != second.unit
    preference = record.preferences[first.preference]
    assert isinstance(preference.weights, priors.PolynomialWeightVector)
    assert preference.hyperparameters is not None
    assert len(preference.hyperparameters.draws) == 2
    assert preference.weights.decay == preference.hyperparameters.draws[0].values[0]
    assert preference.weights.log_noise_scale == preference.hyperparameters.draws[1].values[0]
    assert sum(preference.probabilities) == pytest.approx(1.0)
    assert record.parameter_count == 36
    assert (
        priors.categorical_preferences_from_dict(json.loads(json.dumps(asdict(record)))) == record
    )
    refreshed = priors.CategoricalSampler(
        np.random.default_rng(2802), default_mode="meta", policy="refresh"
    )
    a, b = refreshed.categorical(request), refreshed.categorical(request)
    fresh_record = refreshed.freeze()
    assert a == first
    assert a.preference != b.preference
    assert (
        fresh_record.preferences[a.preference].weights
        != fresh_record.preferences[b.preference].weights
    )
    assert fresh_record.parameter_count == 66
    assert (
        priors.categorical_preferences_from_dict(json.loads(json.dumps(asdict(fresh_record))))
        == fresh_record
    )
    changed_policy = json.loads(json.dumps(asdict(fresh_record)))
    changed_policy["policy"] = "shared"
    with pytest.raises(ValueError, match="categorical.*scope"):
        priors.categorical_preferences_from_dict(changed_policy)
    altered_weights = json.loads(json.dumps(asdict(record)))
    altered_weights["preferences"][0]["weights"]["decay"] = 0.0
    with pytest.raises(ValueError, match="hyperparameter"):
        priors.categorical_preferences_from_dict(altered_weights)
    replay = priors.CategoricalSampler(np.random.default_rng(2802), policy="refresh")
    for draw in fresh_record.draws:
        assert replay.categorical(draw.request) == draw
