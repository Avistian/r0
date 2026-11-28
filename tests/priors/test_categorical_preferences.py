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


def test_categorical_admission_counts_nested_beta_draws_before_using_randomness() -> None:
    request = priors.CategoricalParameter("matrix", ("gaussian", "kernel", "factorized"))
    rng = np.random.default_rng(2803)
    before = json.dumps(rng.bit_generator.state, sort_keys=True)
    for limits, reason in (
        (priors.PreferenceLimits(max_draws=2), "draw limit"),
        (priors.PreferenceLimits(max_values=2), "value limit"),
        (priors.PreferenceLimits(max_parameters=32), "parameter limit"),
    ):
        sampler = priors.CategoricalSampler(rng, limits=limits)
        with pytest.raises(ValueError, match=reason):
            sampler.categorical(request)
        assert json.dumps(rng.bit_generator.state, sort_keys=True) == before
        assert sampler.freeze().draws == ()
    admitted = priors.CategoricalSampler(
        rng, limits=priors.PreferenceLimits(max_draws=4, max_values=4, max_parameters=36)
    )
    admitted.categorical(request)
    admitted.categorical(request)
    record = admitted.freeze()
    assert record.draw_count == 4
    assert record.value_count == 4
    assert record.parameter_count == 36
    encoded = json.loads(json.dumps(asdict(record)))
    assert priors.categorical_preferences_from_dict(encoded) == record
    encoded["limits"]["max_draws"] = 3
    with pytest.raises(ValueError, match="draw limit"):
        priors.categorical_preferences_from_dict(encoded)
    before = json.dumps(rng.bit_generator.state, sort_keys=True)
    with pytest.raises(ValueError, match="draw limit"):
        admitted.categorical(request)
    assert json.dumps(rng.bit_generator.state, sort_keys=True) == before
    assert admitted.freeze() == record


def test_positive_subnormal_mass_survives_probability_display_underflow() -> None:
    import math

    low = 0.07213475204444818
    decay = priors.NumericalParameter(
        "simple_random_weights_decay_rate",
        low,
        6.0,
        shape=(1,),
        scale="log",
        boundary_mass=True,
    )
    scale = priors.NumericalParameter(
        "simple_random_weights_std_scale", 0.0001, 10.0, shape=(1,), scale="log"
    )
    hyper = priors.ParameterPreferences(
        preferences=(
            priors.NumericalPreference(decay, mode="meta", mean=0.5, concentration=1.0),
            priors.NumericalPreference(scale, mode="meta", mean=0.5, concentration=1.0),
        ),
        draws=(
            priors.NumericalDraw(decay, 0, (0.0,), (low,)),
            priors.NumericalDraw(scale, 1, (1.0,), (10.0,)),
        ),
    )
    weights = priors.PolynomialWeightVector(
        (0.0, -74.495, 0.007924812503605782),
        decay=low,
        log_noise_scale=10.0,
        permutation=(0, 1, 2),
    )
    assert weights.values() == (1.2247448713915892, 5e-324, 1.2247448713915892)
    request = priors.CategoricalParameter("tiny", ("left", "middle", "right"))
    preference = priors.CategoricalPreference(
        request, "meta", weights=weights, hyperparameters=hyper
    )
    # Equal outside masses put the midpoint strictly inside the positive middle bin.
    assert preference.choose(0.5) == 1
    assert math.isfinite(preference.log_probabilities[1])
    assert preference.log_probabilities[1] < math.log(5e-324)
    with pytest.raises(ValueError, match="probabilit.*underflow"):
        _ = preference.probabilities
    record = priors.CategoricalPreferences(
        preferences=(preference,), draws=(priors.CategoricalDraw(request, 0, 0.5, 1, "middle"),)
    )
    assert (
        priors.categorical_preferences_from_dict(json.loads(json.dumps(asdict(record)))) == record
    )
