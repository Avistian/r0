"""Standard activation converters preserve explicit state/readout variants."""

import json
from dataclasses import asdict, replace

import numpy as np
import pytest
import torch

from r0.modules import priors


@pytest.mark.parametrize("variant", ["historical-tuple", "historical-graph", "readout-only"])
def test_standard_activation_converter_matches_the_existing_frozen_function(variant: str) -> None:
    transform = priors.calibrate_source_activation(
        priors.CausalActivation("square"),
        np.array([[-2.0], [-1.0], [1.0], [2.0]]),
        rng=np.random.default_rng(128),
        scale=1.0,
        calibration_id="independent-activation-converter",
    )
    base = priors.CausalRecipe(
        task=priors.LinearRecipe(
            rows=16, context_rows=8, features=1, signal_features=1, active_features=1
        ),
        nodes=(
            priors.RootInputCausalNode(noise_scale=(0.0,)),
            priors.ActivatedCausalNode(
                parents=(0,), weights=((2.0,),), noise_scale=(0.0,), noise_position="after_output"
            ),
        ),
        observed=((0, 0),),
        target=((1, 0),),
        version="causal-v2",
    )
    raw = priors.generate_task(base, seed=28, task_id=22)
    reference = priors.generate_task(
        replace(base, nodes=(replace(base.nodes[0], output_activation=transform), base.nodes[1])),
        seed=28,
        task_id=22,
    )
    recipe = priors.CausalPipelineRecipe(
        task=base.task,
        nodes=base.nodes,
        observed=base.observed,
        target=base.target,
        version="causal-v2",
        pipelines=(
            priors.CausalNodePipeline(
                channels=1,
                converters=(
                    priors.NumericalConverter(channel=0, transform=transform, variant=variant),
                ),
            ),
            priors.CausalNodePipeline(channels=1),
        ),
    )
    restored = priors.recipe_from_dict(json.loads(json.dumps(asdict(recipe))))
    actual = priors.generate_task(restored, seed=28, task_id=22)
    torch.testing.assert_close(
        actual.inputs,
        raw.inputs if variant == "historical-tuple" else reference.inputs,
        rtol=0,
        atol=0,
    )
    torch.testing.assert_close(
        actual.scores, raw.scores if variant == "readout-only" else reference.scores, rtol=0, atol=0
    )
    assert priors.explain_generation(actual, [0, 15])["target_max_error"] < 1e-12


@pytest.mark.parametrize("variant", ["historical-tuple", "historical-graph", "readout-only"])
def test_constant_converter_removes_only_its_declared_state_or_readout_path(variant: str) -> None:
    transform = priors.calibrate_source_activation(
        priors.CausalActivation("softmax"),
        np.array([[-2.0], [2.0]]),
        rng=np.random.default_rng(129),
        scale=1.0,
        calibration_id="constant-converter",
    )
    recipe = priors.CausalPipelineRecipe(
        task=priors.LinearRecipe(
            rows=16, context_rows=8, features=1, signal_features=1, active_features=1
        ),
        nodes=(
            priors.RootInputCausalNode(
                bias=(0.0, 0.0),
                root_scale=(1.0, 1.0),
                noise_scale=(0.0, 0.0),
                root_exogenous=priors.ExogenousDistribution(factor=((1.0, 1.0), (0.0, 0.0))),
            ),
            priors.ActivatedCausalNode(
                parents=(0,),
                weights=((1.0,), (0.0,)),
                noise_scale=(0.0,),
                noise_position="after_output",
            ),
        ),
        pipelines=(
            priors.CausalNodePipeline(
                channels=2,
                converters=(
                    priors.NumericalConverter(channel=0, transform=transform, variant=variant),
                ),
            ),
            priors.CausalNodePipeline(channels=1),
        ),
        observed=((0, 1),),
        target=((1, 0),),
        version="causal-v2",
    )
    task = priors.generate_task(recipe, seed=28, task_id=23)
    assert json.loads(task.mechanism_json)["active"] == ([0] if variant == "readout-only" else [])
    if variant != "readout-only":
        assert torch.count_nonzero(task.scores) == 0
    captured = priors.generate_task(replace(recipe, target=((0, 0),)), seed=28, task_id=23)
    assert json.loads(captured.mechanism_json)["active"] == (
        [0] if variant == "historical-tuple" else []
    )
    if variant != "historical-tuple":
        assert torch.count_nonzero(captured.scores) == 0


@pytest.mark.parametrize(("df", "message"), [(1.5, "finite input"), (3.0, "finite output")])
def test_converter_fitting_contract_requires_both_input_and_output_second_moments(
    df: float, message: str
) -> None:
    transform = priors.calibrate_source_activation(
        priors.CausalActivation("square"),
        np.array([[-2.0], [-1.0], [1.0], [2.0]]),
        rng=np.random.default_rng(130),
        scale=1.0,
        calibration_id="finite-sample-not-moment-proof",
    )
    base = priors.CausalPipelineRecipe(
        task=priors.LinearRecipe(
            rows=16, context_rows=8, features=1, signal_features=1, active_features=1
        ),
        nodes=(
            priors.RootInputCausalNode(noise_scale=(0.0,)),
            priors.ActivatedCausalNode(
                parents=(0,), weights=((1.0,),), noise_scale=(0.0,), noise_position="after_output"
            ),
        ),
        pipelines=(
            priors.CausalNodePipeline(
                channels=1, converters=(priors.NumericalConverter(channel=0, transform=transform),)
            ),
            priors.CausalNodePipeline(channels=1),
        ),
        observed=((0, 0),),
        target=((1, 0),),
        version="causal-v2",
    )
    root = replace(
        base.nodes[0],
        root_exogenous=priors.ExogenousDistribution(
            inputs=priors.InputDistribution(distribution="student_t", df=df)
        ),
    )
    with pytest.raises(ValueError, match=message):
        replace(base, nodes=(root, base.nodes[1]))
    bounded = replace(root, output_activation=priors.CausalActivation("tanh"))
    task = priors.generate_task(replace(base, nodes=(bounded, base.nodes[1])), seed=28, task_id=24)
    assert json.loads(task.mechanism_json)["moment_contract"]["target_orders"] == ["all"]
