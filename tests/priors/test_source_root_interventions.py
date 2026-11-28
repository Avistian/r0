"""Versioned source-root replacements preserve descendant innovation coupling."""

import json
from dataclasses import asdict

import pytest
import torch

from r0.modules import priors as p


def base_graph():
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


@pytest.mark.parametrize(
    "coordinate",
    [
        p.SourceRootCoordinate("categorical", probabilities=(0.2, 0.3, 0.5)),
        p.SourceRootCoordinate("zipf", exponent=2.5, clip=10),
        p.SourceRootCoordinate("gaussian", location=3.0, scale=0.5),
    ],
)
def test_source_root_replacement_roundtrip_trace_and_noise_coupling(
    coordinate: p.SourceRootCoordinate,
):
    base = base_graph()
    change = p.StageIntervention(
        0,
        root=p.SourceRootBase(p.SourceRootLaw((coordinate,))),
        version="stage-causal-intervention-v2",
    )
    pair = p.generate_causal_pair(base, interventions=(change,), seed=17, task_id=3)
    torch.testing.assert_close(
        pair.before.targets - 2 * pair.before.inputs[:, 0],
        pair.after.targets - 2 * pair.after.inputs[:, 0],
    )
    raw = json.loads(pair.after.recipe_json)["recipe"]
    restored = p.recipe_from_dict(raw)
    assert isinstance(restored, p.CausalInterventionRecipe)
    assert restored.interventions == (change,)
    assert json.loads(pair.before.recipe_json)["recipe"] == json.loads(json.dumps(asdict(base)))
    trace = p.explain_generation(pair.after, [15, 0, 15])
    target = next(n for n in trace["nodes"] if n["id"] == "targets")
    torch.testing.assert_close(
        torch.tensor(target["values"], dtype=torch.float64).flatten(),
        pair.after.targets[[15, 0, 15]].flatten(),
    )


def test_old_stage_version_rejects_new_root_schema():
    with pytest.raises(ValueError, match="stage"):
        p.StageIntervention(
            0, root=p.SourceRootBase(p.SourceRootLaw((p.SourceRootCoordinate("gaussian"),)))
        )


def test_calibrated_source_replacement_and_invalid_geometry():
    law = p.SourceRootLaw((p.SourceRootCoordinate("zipf", exponent=2.5, clip=10),))
    calibration = p.SourceRootCalibration(law, ("standardize",), 32, (2.0,), (4.0,))
    change = p.StageIntervention(
        0, root=p.SourceRootBase(law, calibration), version="stage-causal-intervention-v2"
    )
    pair = p.generate_causal_pair(base_graph(), interventions=(change,), seed=17, task_id=3)
    assert torch.all(pair.after.inputs >= -0.25)
    assert torch.all(pair.after.inputs <= 2.0)
    decoded = p.recipe_from_dict(json.loads(pair.after.recipe_json)["recipe"])
    assert isinstance(decoded, p.CausalInterventionRecipe)
    assert decoded.interventions == (change,)
    wrong = p.StageIntervention(
        0,
        root=p.SourceRootBase(p.SourceRootLaw(law.coordinates * 2)),
        version="stage-causal-intervention-v2",
    )
    with pytest.raises(ValueError, match="width"):
        p.CausalInterventionRecipe(base_graph(), (wrong,))
