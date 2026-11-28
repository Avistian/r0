"""Source activation decisions retain category multiplicity and replay provenance."""

import json
from dataclasses import asdict

import numpy as np
import pytest

from r0.modules import priors


def test_plain_source_selector_retains_duplicate_slots_and_bound_primitive() -> None:
    # Seed 0 starts with uniform units .6369616873 and .2697867138:
    # plain slot 7 of 12 is fixed-7; primitive 7 of 27 is softplus.
    sampler = priors.SourceActivationSampler(np.random.default_rng(0), mode="local")
    activation = sampler.sample("plain")
    assert activation == priors.CausalActivation("softplus")
    record = sampler.freeze()
    assert record.selections[0].selector == "plain"
    assert record.categorical.draws[0].request.choices == (
        "fixed-0",
        "fixed-1",
        "fixed-2",
        "fixed-3",
        "fixed-4",
        "fixed-5",
        "fixed-6",
        "fixed-7",
        "relu_power",
        "signed_power",
        "reciprocal_abs_power",
        "integer_power",
    )
    assert record.categorical.draws[0].value == "fixed-7"
    assert record.categorical.draws[1].value == "softplus"
    encoded = json.loads(json.dumps(asdict(record)))
    assert priors.source_activation_selections_from_dict(encoded) == record
    encoded["selections"][0]["activation"]["kind"] = "relu"
    with pytest.raises(ValueError, match="selection"):
        priors.source_activation_selections_from_dict(encoded)


def test_rejected_hierarchical_selection_keeps_rng_and_retained_record_unchanged() -> None:
    rng = np.random.default_rng(0)
    sampler = priors.SourceActivationSampler(
        rng, mode="local", categorical_limits=priors.PreferenceLimits(max_draws=1)
    )
    before_rng = json.dumps(rng.bit_generator.state, sort_keys=True)
    before_record = sampler.freeze()
    # The first selector draw fits, but its fixed-primitive draw would exceed the limit.
    with pytest.raises(ValueError, match="draw limit"):
        sampler.sample("plain")
    assert json.dumps(rng.bit_generator.state, sort_keys=True) == before_rng
    assert sampler.freeze() == before_record
