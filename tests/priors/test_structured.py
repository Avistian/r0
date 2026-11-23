import json
import math

import pytest

from r0.modules.priors import Proposal, sample_proposal


def test_scm_records_hidden_causes_and_independent_node_equations() -> None:
    task = sample_proposal(Proposal("scm"))
    parameters = json.loads(task.mechanism_json)["parameters"]
    trace = json.loads(task.trace_json)
    nodes = trace["nodes"][0]
    noise = trace["node_noise"][0]
    for i, parents in enumerate(parameters["parents"]):
        expected = (
            noise[i]
            if not parents
            else math.tanh(
                sum(parameters["weights"][i][j] * nodes[parent] for j, parent in enumerate(parents))
            )
            + noise[i]
        )
        assert nodes[i] == pytest.approx(expected)
    assert trace["latent"][0] == nodes[parameters["target_node"]]
    assert len(task.episode.features) == task.proposal.features
    assert not hasattr(task.episode.predictor_view(), "trace_json")


def test_tree_path_matches_leaf_value() -> None:
    task = sample_proposal(Proposal("tree"))
    parameters = json.loads(task.mechanism_json)["parameters"]
    node = 0
    row = task.source_rows[0]
    for _ in range(parameters["depth"]):
        x = row.values[int(parameters["split_features"][node])]
        assert isinstance(x, (int, float))
        node = 2 * node + (1 if x <= parameters["thresholds"][node] else 2)
    expected = parameters["leaves"][node - len(parameters["split_features"])]
    assert json.loads(task.trace_json)["latent"][0] == expected
