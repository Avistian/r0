"""Soft stage changes preserve original mechanisms and unaffected innovations."""

import json
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

import pytest
import torch

from r0.modules import priors as p


def graph():
    return p.FunctionalCausalRecipe(
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


def test_pipeline_replacement_and_removal_keep_matched_innovations():
    base = graph()
    pipe = p.CausalNodePipeline(1, node_importance=3.0)
    pair = p.generate_causal_pair(
        base,
        interventions=(p.StageIntervention(1, replace_pipeline=True, pipeline=pipe),),
        seed=17,
        task_id=3,
    )
    torch.testing.assert_close(pair.after.targets, 3 * pair.before.targets)
    frozen = replace(base, nodes=(base.nodes[0], replace(base.nodes[1], pipeline=pipe)))
    removed = p.generate_causal_pair(
        frozen, interventions=(p.StageIntervention(1, replace_pipeline=True),), seed=17, task_id=3
    )
    torch.testing.assert_close(removed.after.targets, removed.before.targets / 3)
    raw = json.loads(pair.after.recipe_json)["recipe"]
    decoded = p.recipe_from_dict(raw)
    assert isinstance(decoded, p.CausalInterventionRecipe)
    assert decoded.base == base
    assert raw["base"] == json.loads(json.dumps(asdict(base)))


def test_root_law_change_retains_descendant_noise_and_stream_initial_identity():
    pair = p.generate_causal_pair(
        graph(),
        interventions=(p.StageIntervention(0, root=p.JointRootBase("uniform", 1)),),
        seed=17,
        task_id=3,
    )
    assert torch.all(pair.after.inputs.abs() <= 1)
    assert not torch.equal(pair.before.inputs, pair.after.inputs)
    torch.testing.assert_close(
        pair.before.targets - 2 * pair.before.inputs[:, 0],
        pair.after.targets - 2 * pair.after.inputs[:, 0],
    )
    card = json.loads(pair.card_json)
    assert "0:root" not in card["unaffected_exogenous_equal"]
    assert all(card["stream_initial_equal"].values())


def test_native_emission_change_preserves_vocabulary_and_parent_geometry():
    root = p.NativeCausalNode(
        p.FunctionalCausalNode(
            (), p.FrozenLinearFunction(((1.0,), (-1.0,))), p.JointRootBase("gaussian", 1)
        ),
        p.VariableDomain("binary", ("no", "yes"), "softmax"),
    )
    target = p.NativeCausalNode(
        p.FunctionalCausalNode((0,), p.FrozenLinearFunction(((1.0, -1.0), (-1.0, 1.0)))),
        p.VariableDomain("binary", ("no", "yes"), "softmax"),
    )
    task = p.LinearRecipe(
        rows=16,
        context_rows=8,
        features=1,
        signal_features=1,
        active_features=1,
        task_type="binary",
        classes=2,
    )
    base = p.NativeCausalRecipe(task, (root, target), ((0, 0),), (1, 0))
    change = p.StageIntervention(1, domain=replace(target.domain, emission="argmax"))
    pair = p.generate_causal_pair(base, interventions=(change,), seed=17, task_id=3)
    torch.testing.assert_close(pair.after.targets, pair.before.inputs[:, 0].long())
    decoded = p.recipe_from_dict(json.loads(pair.after.recipe_json)["recipe"])
    assert isinstance(decoded, p.CausalInterventionRecipe)
    assert decoded.base == base
    with pytest.raises(ValueError, match="vocabulary|geometry|domain"):
        p.CausalInterventionRecipe(
            base, (p.StageIntervention(1, domain=replace(target.domain, vocabulary=("yes", "no"))),)
        )


def test_stage_validation_rejects_implicit_changes_and_nonroot_laws():
    with pytest.raises(ValueError, match="stage|root-law"):
        p.StageIntervention(0)
    with pytest.raises(ValueError, match="stage|root-law"):
        p.StageIntervention(0, pipeline=p.CausalNodePipeline(1))
    with pytest.raises(ValueError, match="stage|root-law"):
        p.CausalInterventionRecipe(
            graph(), (p.StageIntervention(1, root=p.JointRootBase("uniform", 1)),)
        )


@pytest.mark.parametrize("seed", [0, 7, 9007199254741011])
def test_combined_stages_roundtrip_with_fixed_measurements(seed: int, tmp_path: Path):
    from r0.modules.generation import load_causal_pair, save_causal_pair

    pipe = p.CausalNodePipeline(1, node_importance=2.0)
    base = graph()
    change = p.StageIntervention(
        0, root=p.JointRootBase("uniform", 1), replace_pipeline=True, pipeline=pipe
    )
    pair = p.generate_causal_pair(
        base, interventions=(change,), seed=seed, task_id=9007199254740993
    )
    torch.testing.assert_close(
        pair.after.targets - 2 * pair.after.inputs[:, 0],
        pair.before.targets - 2 * pair.before.inputs[:, 0],
    )
    assert torch.all(pair.after.inputs.abs() <= 2)
    save_causal_pair(tmp_path / "pair", pair, max_bytes=2_000_000)
    replay = load_causal_pair(tmp_path / "pair")
    torch.testing.assert_close(replay.after.targets, pair.after.targets, rtol=0, atol=0)
    assert p.explain_generation(replay.after, [15, 0, 15])["rows"] == [15, 0, 15]


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("replace_pipeline", 1),
        ("version", "unknown"),
        ("domain", {"kind": "continuous"}),
        ("root", {"family": "uniform"}),
    ],
)
def test_stage_malformed_objects_reject(field: str, value: Any):
    fields: dict[str, Any] = {"node": 0, "replace_pipeline": True, field: value}
    with pytest.raises(ValueError, match="stage"):
        p.StageIntervention(**fields)


def test_stage_archive_and_geometry_rejections():
    base = graph()
    change = p.StageIntervention(0, root=p.JointRootBase("uniform", 1))
    recipe = p.CausalInterventionRecipe(base, (change,))
    raw = json.loads(json.dumps(asdict(recipe)))
    raw["interventions"][0]["root"]["width"] = 2
    with pytest.raises(ValueError, match="root|width|geometry"):
        p.recipe_from_dict(raw)
    with pytest.raises(ValueError, match="overlap"):
        p.CausalInterventionRecipe(base, (change, p.HardIntervention(0, (0,), (2.0,))))
    with pytest.raises(ValueError, match="domain"):
        p.CausalInterventionRecipe(base, (p.StageIntervention(0, domain=p.VariableDomain()),))
