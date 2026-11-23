from dataclasses import replace
from typing import cast

import pytest

from r0.modules.priors import Proposal, sample_proposal


def test_sparse_proposal_replays_and_pairs_noise_without_changing_mechanism() -> None:
    proposal = Proposal("linear", seed=11)
    clean = sample_proposal(proposal)
    noisy = sample_proposal(replace(proposal, noise="added"))
    assert sample_proposal(proposal) == clean
    assert noisy.mechanism_json == clean.mechanism_json
    assert noisy.episode.context == clean.episode.context
    assert noisy.episode.query_targets != clean.episode.query_targets
    assert sample_proposal(replace(proposal, observation=1)).mechanism_json == clean.mechanism_json
    assert clean.episode.predictor_view().context_targets == clean.episode.context_targets
    with pytest.raises(ValueError, match="limit"):
        Proposal("linear", rows=129)


def test_interaction_rules_retain_independent_construction_trace() -> None:
    import json
    import math

    for draw in range(3):
        task = sample_proposal(Proposal("interaction", draw=draw))
        parameters = json.loads(task.mechanism_json)["parameters"]
        latent = json.loads(task.trace_json)["latent"]
        degree = parameters["degree"]
        for row, value in zip(task.source_rows, latent, strict=True):
            inputs = [float(cast(float, v)) for v in row.values[:degree]]
            if parameters["rule"] == "product":
                expected = math.prod(inputs)
            elif parameters["rule"] == "parity":
                expected = float(sum(v > 0 for v in inputs) % 2)
            else:
                expected = math.prod(inputs[1:]) if inputs[0] > 0 else -1.0
            assert value == pytest.approx(expected)
