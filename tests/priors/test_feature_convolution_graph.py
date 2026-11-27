"""Shared feature kernels compose as causal equations without dense expansion."""

import json
from dataclasses import asdict, replace
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
import torch

from r0.modules import priors as p


def convolution():
    return p.FrozenFeatureConv1D(
        4,
        1,
        (p.FeatureConv1DLayer(p.Conv1DGeometry(1, 2), (((2.0, -1.0),),), (0.5,)),),
    )


def test_frozen_interface_and_parent_composition():
    function = convolution()
    inputs = torch.arange(16, dtype=torch.float64).reshape(4, 4)
    expected = 2 * inputs[:, :-1] - inputs[:, 1:] + 0.5
    assert torch.equal(p.evaluate_frozen_function(function, inputs), expected)
    decoded = p.frozen_function_from_dict(json.loads(json.dumps(asdict(function))))
    assert decoded == function
    parent = p.FrozenParentFunction((function, function), aggregation="mean")
    combined = torch.cat((inputs, inputs + 2), dim=1)
    assert torch.equal(p.evaluate_frozen_function(parent, combined), expected + 1)


def test_causal_convolution_channels_and_replay():
    root = p.FunctionalCausalNode(
        (),
        p.FrozenLinearFunction(tuple(tuple(float(i == j) for j in range(4)) for i in range(4))),
        root=p.JointRootBase("gaussian", 4),
    )
    node = p.FunctionalCausalNode((0,), convolution())
    task = p.LinearRecipe(rows=16, context_rows=8, features=2, signal_features=2, active_features=2)
    recipe = p.FunctionalCausalRecipe(task, (root, node), ((0, 0), (0, 1)), ((1, 0),))
    saved = p.generate_task(recipe, seed=17, task_id=2**54 + 9)
    np.testing.assert_allclose(
        saved.scores[:, 0], 2 * saved.inputs[:, 0] - saved.inputs[:, 1] + 0.5
    )
    decoded = p.recipe_from_dict(json.loads(saved.recipe_json)["recipe"])
    assert torch.equal(p.generate_task(decoded, seed=17, task_id=2**54 + 9).targets, saved.targets)
    assert p.explain_generation(saved, [15, 0, 15])["signal_max_error"] == 0


def test_convolution_moments_track_active_taps_and_bounded_activation():
    root = p.FunctionalCausalNode(
        (),
        p.FrozenLinearFunction(((0.0,), (0.0,), (0.0,), (0.0,))),
        root=p.JointRootBase("gaussian", 1),
        noise_scale=(1.0, 1.0, 1.0, 1.0),
        exogenous=p.ExogenousDistribution(p.InputDistribution("student_t", df=1.5)),
    )
    function = convolution()
    task = p.LinearRecipe(rows=16, context_rows=8, features=1, signal_features=1, active_features=1)
    with pytest.raises(ValueError, match="second moments"):
        p.CalibratedFunctionalCausalRecipe(
            task,
            (root, p.FunctionalCausalNode((0,), function)),
            ((0, 0),),
            ((1, 0),),
            calibration_rows=32,
        )
    bounded = replace(
        function, layers=(replace(function.layers[0], activation=p.CausalActivation("tanh")),)
    )
    recipe = p.CalibratedFunctionalCausalRecipe(
        task,
        (root, p.FunctionalCausalNode((0,), bounded)),
        ((0, 0),),
        ((1, 0),),
        calibration_rows=32,
    )
    saved = p.generate_task(recipe, seed=3, task_id=8)
    assert torch.isfinite(saved.scores).all()
    disconnected = replace(function, layers=(replace(function.layers[0], kernel=(((0.0, 0.0),),)),))
    p.CalibratedFunctionalCausalRecipe(
        task,
        (root, p.FunctionalCausalNode((0,), disconnected)),
        ((0, 0),),
        ((1, 0),),
        calibration_rows=32,
    )


def test_shared_convolution_admission_precedes_backend_and_finite_mask():
    function = convolution()
    huge = replace(
        function,
        layers=(
            replace(
                function.layers[0],
                geometry=p.Conv1DGeometry(1, 2, dilation=2**30, padding="same-upper"),
            ),
        ),
    )
    with (
        patch("torch.isfinite", side_effect=AssertionError("mask before admission")),
        pytest.raises(ValueError, match="memory"),
    ):
        p.evaluate_frozen_function(huge, torch.zeros((1, 4), dtype=torch.float64))
    raw = json.loads(json.dumps(asdict(function)))
    del raw["layers"][0]["activation_scope"]
    with pytest.raises(ValueError, match="every declared field"):
        p.frozen_function_from_dict(raw)


@pytest.mark.parametrize("layout", ["channel-major", "position-major"])
def test_grouped_convolution_intervention_archive_and_channel_ancestry(layout: str, tmp_path: Path):
    from r0.modules import generation as g

    # Two channels, two positions: grouped kernels must never cross channels.
    function = p.FrozenFeatureConv1D(
        2,
        2,
        (
            p.FeatureConv1DLayer(
                p.Conv1DGeometry(2, 1, groups=2), (((2.0,),), ((-3.0,),)), (1.0, 2.0)
            ),
        ),
        layout,
    )
    root = p.FunctionalCausalNode(
        (),
        p.FrozenLinearFunction(tuple(tuple(float(i == j) for j in range(4)) for i in range(4))),
        root=p.JointRootBase("gaussian", 4),
    )
    task = p.LinearRecipe(rows=16, context_rows=8, features=4, signal_features=4, active_features=4)
    recipe = p.FunctionalCausalRecipe(
        task,
        (root, p.FunctionalCausalNode((0,), function)),
        tuple((0, j) for j in range(4)),
        ((1, 0),),
    )
    pair = p.generate_causal_pair(
        recipe,
        interventions=(
            p.HardIntervention(0, channels=(0,), values=(4.0,), location="state_and_readout"),
        ),
        seed=23,
        task_id=2**54 + 3,
    )
    np.testing.assert_array_equal(pair.after.scores.numpy(), np.full((16, 1), 9.0))
    untouched = p.generate_causal_pair(
        recipe,
        interventions=(
            p.HardIntervention(0, channels=(3,), values=(17.0,), location="state_and_readout"),
        ),
        seed=23,
        task_id=2**54 + 3,
    )
    assert torch.equal(untouched.before.scores, untouched.after.scores)
    trace = p.explain_generation(pair.after, [15, 0, 15])
    assert trace["rows"] == [15, 0, 15]
    path = tmp_path / "conv.zip"
    g.save_group(path, [(2**54 + 3, pair.after)], max_bytes=2_000_000)
    restored = next(g.load_group(path))
    assert restored.recipe_json == pair.after.recipe_json
    replay = p.generate_task(
        p.recipe_from_dict(json.loads(restored.recipe_json)["recipe"]), seed=23, task_id=2**54 + 3
    )
    assert torch.equal(replay.targets, pair.after.targets)


@pytest.mark.parametrize("wrapped", [False, True])
def test_shared_function_counts_complete_convolution_metadata_before_backend(wrapped: bool):
    layer = p.FeatureConv1DLayer(p.Conv1DGeometry(1, 1), (((1.0,),),), (0.0,))
    function = p.FrozenFeatureConv1D(1, 1, (layer,) * 32)
    member = p.FrozenParentFunction((function,), "mean") if wrapped else function
    with (
        patch("torch.nn.functional.conv1d", side_effect=AssertionError("backend before metadata")),
        pytest.raises(ValueError, match="metadata|parameter"),
    ):
        p.evaluate_frozen_function(
            member, torch.ones((1, 1), dtype=torch.float64), limits=p.CausalLimits(parameters=200)
        )
