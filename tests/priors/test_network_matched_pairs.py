"""Estimated comparisons use one independent population and retain the original archive."""

import json
from dataclasses import replace
from typing import Any, cast

import numpy as np
import pytest
import torch

from r0.modules import priors as p


def network() -> p.NetworkRecipe:
    return p.NetworkRecipe(
        p.LinearRecipe(rows=27, context_rows=11, features=2, signal_features=2, active_features=2),
        p.FrozenNetworkFunction(
            2, (p.NetworkBlock((0,), (p.NetworkAffine(((2.0, -3.0),), (0.5,)),)),)
        ),
        p.JointRootBase("ball", 2),
    )


def test_estimated_pair_literal_replay_trace_and_independent_identity():
    original = network()
    pair = p.generate_pair(
        original,
        intervention="sparsity",
        value=1,
        seed=7,
        task_id=2**54 + 3,
        allow_estimated_variance=True,
    )
    before = p.generate_task(original, seed=7, task_id=2**54 + 3)
    assert pair.before.recipe_json == before.recipe_json
    changed = p.recipe_from_dict(json.loads(pair.after.recipe_json)["recipe"])
    assert isinstance(changed, p.MatchedNetworkInputView)
    matching = changed.matching
    assert matching is not None
    raw = 2 * pair.after.inputs[:, 0].numpy() + 0.5
    expected = (raw - matching.after_mean[0]) / matching.after_std[0] * matching.before_std[
        0
    ] + matching.before_mean[0]
    np.testing.assert_allclose(pair.after.scores[:, 0], expected, atol=1e-14)
    assert torch.equal(pair.before.inputs, pair.after.inputs)
    other = p.generate_pair(
        original,
        intervention="sparsity",
        value=1,
        seed=19,
        task_id=3,
        allow_estimated_variance=True,
    )
    assert p.recipe_from_dict(json.loads(other.after.recipe_json)["recipe"]) == changed
    card = p.compare_pair(
        pair.before, pair.after, intervention="sparsity", allow_estimated_variance=True
    )
    assert card["variance_control"] == "independent_network_baseline_moments_estimate"
    trace = p.explain_generation(pair.after, [26, 0, 26])
    assert trace["rows"] == [26, 0, 26]
    nodes = {n["id"]: n for n in trace["nodes"]}
    assert "view-matching" in nodes
    assert "view-raw-scores" in nodes
    assert nodes["view-matching"]["parents"] == ["view-raw-scores"]
    assert nodes["signal"]["parents"] == ["view-matching"]
    np.testing.assert_allclose(nodes["view-raw-scores"]["values"], raw[[26, 0, 26], None])
    np.testing.assert_allclose(nodes["view-matching"]["values"], expected[[26, 0, 26], None])
    seen: set[str] = set()
    for node in trace["nodes"]:
        assert set(node["parents"]) <= seen
        seen.add(node["id"])
    with pytest.raises(ValueError, match="outside"):
        p.compare_pair(pair.before, pair.after, intervention="sparsity")


def test_matching_preserved_by_nonstructural_pairs_and_rejects_recalibration():
    changed = p.paired_recipe(
        network(), intervention="sparsity", value=1, allow_estimated_variance=True
    )
    assert isinstance(changed, p.MatchedNetworkInputView)
    for intervention, value in (("irrelevant", 5), ("noise", 0.25)):
        pair = p.generate_pair(changed, intervention=intervention, value=value, seed=7, task_id=3)
        restored = p.recipe_from_dict(json.loads(pair.after.recipe_json)["recipe"])
        assert isinstance(restored, p.MatchedNetworkInputView)
        assert restored.matching == changed.matching
    for estimate in (False, True):
        with pytest.raises(ValueError, match="already matched"):
            p.generate_pair(
                changed,
                intervention="sparsity",
                value=2,
                seed=7,
                task_id=3,
                allow_estimated_variance=estimate,
            )


def test_multiclass_separation_preserves_matching_and_limits_fail_closed():
    source = network()
    source = replace(
        source,
        task=replace(source.task, task_type="multiclass", classes=3),
        network=p.FrozenNetworkFunction(
            2,
            (
                p.NetworkBlock(
                    (0,),
                    (p.NetworkAffine(((2.0, -3.0), (-1.0, 2.0), (0.5, 1.0)), (0.1, -0.2, 0.3)),),
                ),
            ),
        ),
    )
    changed = p.paired_recipe(
        source, intervention="sparsity", value=1, allow_estimated_variance=True
    )
    assert isinstance(changed, p.MatchedNetworkInputView)
    pair = p.generate_pair(changed, intervention="separation", value=0.4, seed=7, task_id=3)
    assert torch.equal(pair.before.scores, pair.after.scores)
    assert p.compare_pair(pair.before, pair.after, intervention="separation")
    with pytest.raises(ValueError, match="limit"):
        p.compare_pair(
            pair.before,
            pair.after,
            intervention="separation",
            limits=p.CausalLimits(parameters=100),
        )
    with pytest.raises(ValueError, match="boolean"):
        p.paired_recipe(
            source, intervention="sparsity", value=1, allow_estimated_variance=cast(Any, 1)
        )
