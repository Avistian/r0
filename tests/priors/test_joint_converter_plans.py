"""Converter plans retain target policy and only the selected branch's choices."""

import copy
import json
from dataclasses import asdict

import numpy as np
import pytest
import torch

from r0.modules import priors


def test_joint_target_identity_preserves_shared_converter_catalog_and_rng() -> None:
    rng = np.random.default_rng(0)
    control = copy.deepcopy(rng)
    context = priors.JointSamplingContext(rng, mode="global")
    prior = priors.JointNumericalConverterPrior(disallow_target_warping=True)
    target = priors.sample_joint_converter(
        context, "target", channel=0, feature_group="y", prior=prior
    )
    assert target.mode == "id"
    assert context.freeze().categorical.draws == ()
    assert rng.bit_generator.state == control.bit_generator.state
    plans = [target]
    for name, group in (("feature", "x"), ("near-target", "Y"), ("target-prefix", "y-extra")):
        plans.append(
            priors.sample_joint_converter(
                context, name, channel=0, feature_group=group, prior=prior
            )
        )
    record = context.freeze()
    assert [plan.mode for plan in plans] == ["id", "kumar", "kumar", "kumar"]
    assert len(record.categorical.preferences) == 1
    assert record.categorical.preferences[0].request.choices == ("id", "kumar", "act")
    assert [d.preference for d in record.categorical.draws] == [0, 0, 0]
    assert [d.preference for d in record.numerical.draws] == [0, 1, 0, 1, 0, 1]
    assert record.events[0].reason == "target-warp-disabled"
    for plan in plans:
        plan.validate(record)
        assert (
            priors.joint_converter_plan_from_dict(
                json.loads(json.dumps(asdict(plan))), decisions=record
            )
            == plan
        )


@pytest.mark.parametrize("variant", ["historical-tuple", "historical-graph", "readout-only"])
def test_fitted_joint_converter_drives_declared_state_and_readout_paths(variant: str) -> None:
    context = priors.JointSamplingContext(np.random.default_rng(0), mode="global")
    prior = priors.JointNumericalConverterPrior(allow_activation=False, variant=variant)
    plan = priors.sample_joint_converter(
        context, "feature", channel=0, feature_group="x", prior=prior
    )
    assert plan.mode == "kumar"
    converter = priors.fit_joint_converter(
        plan,
        context.freeze(),
        np.array([[-2.0], [0.0], [2.0]]),
        rng=np.random.default_rng(99),
        calibration_id="independent-converter-fit",
    )
    assert plan.shapes is not None
    a, b = plan.shapes
    warped = 1.0 - (1.0 - 0.625**a) ** b
    recipe = priors.CausalPipelineRecipe(
        task=priors.LinearRecipe(
            rows=16, context_rows=8, features=1, signal_features=1, active_features=1
        ),
        nodes=(
            priors.ActivatedCausalNode(
                bias=(0.5,), noise_scale=(0.0,), noise_position="after_output"
            ),
            priors.ActivatedCausalNode(
                parents=(0,), weights=((4.0,),), noise_scale=(0.0,), noise_position="after_output"
            ),
        ),
        pipelines=(
            priors.CausalNodePipeline(channels=1, converters=(converter,), node_importance=2.0),
            priors.CausalNodePipeline(channels=1),
        ),
        observed=((0, 0),),
        target=((1, 0),),
        version="causal-v2",
    )
    restored = priors.recipe_from_dict(json.loads(json.dumps(asdict(recipe))))
    task = priors.generate_task(restored, seed=28, task_id=22)
    feature = 0.5 if variant == "historical-tuple" else warped
    score = 4.0 if variant == "readout-only" else 8.0 * warped
    torch.testing.assert_close(task.inputs, torch.full((16, 1), feature, dtype=torch.float64))
    torch.testing.assert_close(task.scores, torch.full((16, 1), score, dtype=torch.float64))
    assert priors.explain_generation(task, [0, 15])["target_max_error"] < 1e-12
