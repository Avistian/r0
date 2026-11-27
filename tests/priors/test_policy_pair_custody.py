"""Caller admission never rewrites the retained before/after graph declaration."""

import json
from dataclasses import asdict, replace
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from r0.modules import generation as g
from r0.modules import priors as p


def policy(mode: str):
    base = p.FunctionalCausalRecipe(
        p.LinearRecipe(rows=16, context_rows=8, features=1, active_features=1, signal_features=1),
        (
            p.FunctionalCausalNode(
                (), p.FrozenLinearFunction(((1.0,),)), p.JointRootBase("gaussian", 1)
            ),
            p.FunctionalCausalNode((0,), p.FrozenLinearFunction(((2.0,),))),
        ),
        ((0, 0),),
        ((1, 0),),
    )
    request = p.InterventionPolicyRequest(
        base,
        (1,),
        profile="explicit-policy-r0-v1",
        mode=mode,
        limits=p.CausalLimits(working_bytes=128 * 1024**2),
    )
    return p.sample_intervention_policy(
        request,
        selection_rng=np.random.default_rng(3),
        value_rng=np.random.default_rng(5),
        calibration_rng=np.random.default_rng(7),
    )


@pytest.mark.parametrize("mode", ["matched", "population"])
def test_policy_pair_caller_limits_preserve_original_records(mode: str, tmp_path: Path):
    sampled = policy(mode)
    pair = p.generate_policy_pair(sampled, seed=11, task_id=2**54 + 19)
    original = json.loads(json.dumps(asdict(sampled.request.base)))
    assert json.loads(pair.before.recipe_json)["recipe"] == original
    assert json.loads(pair.after.recipe_json)["recipe"]["request"]["base"] == original
    for cap in (32, 64, 128):
        limits = p.CausalLimits(working_bytes=cap * 1024**2)
        assert p.compare_causal_pair(pair.before, pair.after, limits=limits) == json.loads(
            pair.card_json
        )
        path = tmp_path / str(cap)
        g.save_causal_pair(path, pair, max_bytes=16 * 1024**2, limits=limits)
        reopened = g.load_causal_pair(path, limits=limits)
        assert reopened.before.recipe_json == pair.before.recipe_json
        assert reopened.after.recipe_json == pair.after.recipe_json
    with (
        patch.object(
            type(sampled.initial[0]),
            "restore",
            side_effect=AssertionError("replay before admission"),
        ),
        pytest.raises(ValueError, match="limit|storage|working"),
    ):
        p.compare_causal_pair(pair.before, pair.after, limits=p.CausalLimits(working_bytes=1))
    with pytest.raises(ValueError, match="metadata|semantic"):
        p.compare_causal_pair(
            pair.before, replace(pair.after, source_task_id="altered"), limits=limits
        )
