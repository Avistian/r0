"""Frozen-network input comparisons preserve actual shared draws and source records."""

import json
from dataclasses import replace

import pytest
import torch

from r0.modules import priors as p


def network() -> p.NetworkRecipe:
    return p.NetworkRecipe(
        p.LinearRecipe(rows=32, context_rows=16, features=2, signal_features=2, active_features=2),
        p.FrozenNetworkFunction(
            2, (p.NetworkBlock((0,), (p.NetworkAffine(((2.0, -3.0),), (0.5,)),)),)
        ),
        p.JointRootBase("ball", 2),
    )


def test_irrelevant_pair_preserves_complete_original_record_and_arrays() -> None:
    recipe = network()
    original = p.generate_task(recipe, seed=7, task_id=9007199254740993)
    pair = p.generate_pair(
        recipe, intervention="irrelevant", value=5, seed=7, task_id=9007199254740993
    )
    assert pair.before.recipe_json == original.recipe_json
    assert torch.equal(pair.before.inputs, pair.after.inputs[:, :2])
    assert torch.equal(pair.before.targets, pair.after.targets)
    assert all(json.loads(pair.card_json)["required_equalities"].values())
    assert (
        json.loads(pair.after.recipe_json)["recipe"]["base"]
        == json.loads(original.recipe_json)["recipe"]
    )
    after = p.recipe_from_dict(json.loads(pair.after.recipe_json)["recipe"])
    second = p.generate_pair(
        after, intervention="irrelevant", value=7, seed=7, task_id=9007199254740993
    )
    assert torch.equal(pair.after.inputs, second.after.inputs[:, :5])


def test_input_mask_pair_is_explicitly_uncontrolled_variance() -> None:
    pair = p.generate_pair(network(), intervention="sparsity", value=1, seed=7, task_id=9)
    assert torch.equal(pair.before.inputs, pair.after.inputs)
    assert torch.equal(pair.after.scores[:, 0], 2 * pair.before.inputs[:, 0] + 0.5)
    card = json.loads(pair.card_json)
    assert card["variance_control"] == "network_input_mask_uncontrolled"
    assert card["required_equalities"]["nested_active_features"]
    assert card["required_equalities"]["frozen_network"]
    after = p.recipe_from_dict(json.loads(pair.after.recipe_json)["recipe"])
    restored = p.generate_pair(after, intervention="sparsity", value=2, seed=7, task_id=9)
    assert torch.equal(pair.before.scores, restored.after.scores)


def test_network_pair_rejects_forged_after_values_and_execution_version() -> None:
    pair = p.generate_pair(network(), intervention="irrelevant", value=4, seed=7, task_id=9)
    with pytest.raises(ValueError, match="replay"):
        p.compare_pair(
            pair.before,
            replace(pair.after, scores=pair.after.scores + 1),
            intervention="irrelevant",
        )
    record = json.loads(pair.after.recipe_json)
    record["execution"]["implementation"] = "not-a-network-view"
    with pytest.raises(ValueError, match="implementation"):
        p.compare_pair(
            pair.before,
            replace(pair.after, recipe_json=json.dumps(record)),
            intervention="irrelevant",
        )


def test_network_view_original_row_trace_retains_mask_and_typed_envelope() -> None:
    pair = p.generate_pair(network(), intervention="sparsity", value=1, seed=7, task_id=9)
    graph = p.explain_generation(pair.after, [17, 2, 17, 0])
    assert graph["rows"] == [17, 2, 17, 0]
    assert graph["version"] == "generation-graph-v1"
    assert graph["episode_id"] == pair.after.episode_id
    by_id = {node["id"]: node for node in graph["nodes"]}
    assert "view-mask" in by_id
    assert by_id["network-input"]["parents"] == ["view-mask"]
    assert all(row[1] == 0 for row in by_id["view-mask"]["values"])


@pytest.mark.parametrize(
    "family",
    ["dense", "individual-sparse", "block-sparse", "skip", "residual", "irregular", "gated"],
)
def test_multiclass_sampled_architectures_support_both_raw_comparisons(family: str) -> None:
    import numpy as np

    sampled = p.sample_network(
        p.NetworkSamplingRequest(2, (3, 3), families=(family,)),
        named_rng=np.random.default_rng(18),
        raw_rng=np.random.default_rng(19),
    )
    controls = replace(network().task, task_type="multiclass", classes=3, separation=0.7)
    recipe = p.NetworkRecipe(controls, sampled, p.JointRootBase("ball", 2))
    for intervention, value in (("irrelevant", 4), ("sparsity", 1)):
        pair = p.generate_pair(recipe, intervention=intervention, value=value, seed=8, task_id=11)
        assert all(json.loads(pair.card_json)["required_equalities"].values())
        assert p.compare_pair(pair.before, pair.after, intervention=intervention)
        if intervention == "irrelevant":
            assert pair.before.probabilities is not None
            assert pair.after.probabilities is not None
            assert torch.equal(pair.before.probabilities, pair.after.probabilities)
