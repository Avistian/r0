"""Sampled source roots retain independent selection/calibration in stage replacements."""

import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pytest
import torch

from r0.modules import priors as p
from r0.modules.generation import load_causal_pair, save_causal_pair


def fixture(mode: str):
    profile = p.sample_source_root_profile(
        p.SourceRootProfile(1, selector="forced-r0", families=("gaussian",)),
        model_rng=np.random.default_rng(20),
        selection_rng=np.random.default_rng(21),
    )
    root = (
        p.sample_source_root_base(
            profile,
            modes=(mode,),
            calibration_rows=32,
            calibration_rng=np.random.default_rng(22),
        )
        if mode == "standardize"
        else p.sample_source_root_base(profile, modes=(mode,))
    )
    base = p.FunctionalCausalRecipe(
        p.LinearRecipe(rows=16, context_rows=8, features=1, signal_features=1, active_features=1),
        (
            p.FunctionalCausalNode(
                (), p.FrozenLinearFunction(((1.0,),)), p.JointRootBase("gaussian", 1)
            ),
            p.FunctionalCausalNode((0,), p.FrozenLinearFunction(((2.0,),)), noise_scale=(0.5,)),
        ),
        ((0, 0),),
        ((1, 0),),
    )
    return base, root


@pytest.mark.parametrize("mode", ["identity", "standardize"])
def test_sampled_stage_retains_custody_roundtrip_noise_and_saved_pair(mode: str, tmp_path: Path):
    base, root = fixture(mode)
    change = p.StageIntervention(0, root=root, version="stage-causal-intervention-v2")
    pair = p.generate_causal_pair(base, interventions=(change,), seed=19, task_id=2**54 + 3)
    torch.testing.assert_close(
        pair.before.targets - 2 * pair.before.inputs[:, 0],
        pair.after.targets - 2 * pair.after.inputs[:, 0],
    )
    payload = json.loads(pair.after.recipe_json)["recipe"]
    assert payload["interventions"][0]["root"] == json.loads(json.dumps(asdict(root)))
    restored = p.recipe_from_dict(payload)
    assert isinstance(restored, p.CausalInterventionRecipe)
    assert restored.interventions == (change,)
    assert json.loads(pair.before.recipe_json)["recipe"] == json.loads(json.dumps(asdict(base)))
    save_causal_pair(tmp_path / "pair", pair, max_bytes=4_000_000)
    loaded = load_causal_pair(tmp_path / "pair")
    assert loaded.after.episode() == pair.after.episode()
    trace = p.explain_generation(loaded.after, [15, 0, 15])
    assert trace["rows"] == [15, 0, 15]
    payload["interventions"][0]["root"]["profile"]["source"] = "invented"
    with pytest.raises(ValueError, match="source|profile|root|intervention"):
        p.recipe_from_dict(payload)


def test_sampled_stage_v1_rejects_envelope_and_strict_decode_limits():
    base, root = fixture("standardize")
    with pytest.raises(ValueError, match="stage"):
        p.StageIntervention(0, root=root)
    recipe = p.CausalInterventionRecipe(
        base, (p.StageIntervention(0, root=root, version="stage-causal-intervention-v2"),)
    )
    with pytest.raises(ValueError, match="limit|budget"):
        p.recipe_from_dict(asdict(recipe), limits=p.CausalLimits(working_bytes=1024))
