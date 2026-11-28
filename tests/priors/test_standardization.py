"""Standalone source Standardize fits once and applies its frozen population constants."""

import json
from dataclasses import asdict, replace

import numpy as np
import pytest
import torch

from r0.modules import priors


def test_standardization_preserves_population_scale_and_constant_channel_behavior() -> None:
    sample = np.array([[1.0, 7.0], [3.0, 7.0]])
    fitted = priors.calibrate_standardization(sample, calibration_id="independent-two-row-fit")
    assert fitted.mean == (2.0, 7.0)
    assert fitted.inverse_std == (1 / 1.0001, 10000.0)
    frozen = asdict(fitted)
    value = torch.tensor([[1.0, 7.0], [3.0, 7.5], [5.0, 6.0]], dtype=torch.float64)
    expected = torch.tensor(
        [[-1 / 1.0001, 0.0], [1 / 1.0001, 5000.0], [3 / 1.0001, -10000.0]],
        dtype=torch.float64,
    )
    torch.testing.assert_close(fitted.apply(value), expected, rtol=1e-14, atol=1e-14)
    sample[:] = 123.0
    fitted.apply(value[[2]])
    assert asdict(fitted) == frozen


def test_standardization_precedes_channel_weights_l2_and_readout_capture() -> None:
    standard = priors.calibrate_standardization(
        np.array([[1.0, 7.0], [3.0, 7.0]]), calibration_id="independent-pipeline-standardize"
    )
    l2 = priors.calibrate_l2_normalization(
        np.array([[-2 / 1.0001, 0.0], [2 / 1.0001, 0.0]]),
        calibration_id="independent-pipeline-post-weight-l2",
    )
    recipe = priors.CausalPipelineRecipe(
        task=priors.LinearRecipe(
            rows=16, context_rows=8, features=2, signal_features=2, active_features=2
        ),
        nodes=(
            priors.RootInputCausalNode(
                bias=(1.0, 7.0),
                root_scale=(1.0, 0.0),
                noise_scale=(0.3, 0.4),
            ),
            priors.ActivatedCausalNode(
                parents=(0,),
                weights=((1.0,), (-2.0,)),
                noise_scale=(0.0,),
                noise_position="after_output",
            ),
        ),
        pipelines=(
            priors.CausalNodePipeline(
                channels=2,
                stages=(standard,),
                channel_importance=(2.0, -1.0),
                l2=l2,
                converters=(priors.NumericalConverter(channel=0),),
                node_importance=3.0,
            ),
            priors.CausalNodePipeline(channels=1),
        ),
        observed=((0, 0), (0, 1)),
        target=((1, 0),),
        version="causal-v2",
    )
    restored = priors.recipe_from_dict(json.loads(json.dumps(asdict(recipe))))
    assert restored == recipe
    task = priors.generate_task(restored, seed=28, task_id=26)
    steps = {node["id"]: node for node in priors.explain_generation(task, [0, 15])["nodes"]}
    root = torch.tensor(steps["root-input-0"]["values"], dtype=torch.float64)
    noise = torch.tensor(steps["innovation-0"]["values"], dtype=torch.float64)
    first = 2 * (root[:, 0] - 1) / 1.0001 / (2 / 1.0001 + 1e-8)
    torch.testing.assert_close(task.inputs[[0, 15], 0], first, rtol=1e-12, atol=1e-12)
    torch.testing.assert_close(
        task.scores[[0, 15], 0], 3 * (first + noise[:, 0] - 2 * noise[:, 1]), rtol=1e-12, atol=1e-12
    )
    assert (
        steps["pipeline-0-stage-0"]["parameters"]["pipeline"]["stages"][0]["kind"] == "standardize"
    )


def test_standardization_requires_population_second_moments_at_its_actual_position() -> None:
    standard = priors.calibrate_standardization(
        np.array([[-1.0, -2.0], [1.0, 2.0]]), calibration_id="finite-sample-is-not-a-moment-proof"
    )
    root = priors.RootInputCausalNode(
        bias=(0.0, 0.0),
        root_scale=(1.0, 1.0),
        noise_scale=(0.0, 0.0),
        root_exogenous=priors.ExogenousDistribution(
            inputs=priors.InputDistribution(distribution="student_t", df=1.5)
        ),
    )
    recipe = priors.CausalPipelineRecipe(
        task=priors.LinearRecipe(
            rows=16, context_rows=8, features=1, signal_features=1, active_features=1
        ),
        nodes=(root,),
        pipelines=(priors.CausalNodePipeline(channels=2),),
        observed=((0, 0),),
        target=((0, 1),),
        version="causal-v2",
    )
    pipeline = priors.CausalNodePipeline(channels=2, stages=(standard,))
    with pytest.raises(ValueError, match="finite input second moments"):
        replace(recipe, pipelines=(pipeline,))
    # A later zero multiplier cannot justify fitting an infinite-variance input.
    with pytest.raises(ValueError, match="finite input second moments"):
        replace(recipe, pipelines=(replace(pipeline, channel_importance=(0.0, 0.0)),))
    bounded = replace(root, output_activation=priors.CausalActivation("tanh"))
    task = priors.generate_task(
        replace(recipe, nodes=(bounded,), pipelines=(pipeline,)), seed=28, task_id=27
    )
    assert json.loads(task.mechanism_json)["moment_contract"]["target_orders"] == ["all"]


@pytest.mark.parametrize("placement", ["input", "output"])
@pytest.mark.parametrize("root_input", [False, True])
def test_standardization_in_node_equations_replays_and_checks_its_input_law(
    placement: str, root_input: bool
) -> None:
    standard = priors.calibrate_standardization(
        np.array([[1.0], [3.0]]), calibration_id="independent-node-standardize"
    )
    node = priors.ActivatedCausalNode(parents=(0,), weights=((2.0,),), noise_scale=(0.0,))
    node = replace(node, **{placement + "_activation": standard})
    root = priors.RootInputCausalNode(noise_scale=(0.0,)) if root_input else priors.CausalNode()
    recipe = priors.CausalRecipe(
        task=priors.LinearRecipe(
            rows=16, context_rows=8, features=1, signal_features=1, active_features=1
        ),
        nodes=(root, node),
        observed=((0, 0),),
        target=((1, 0),),
        version="causal-v2",
    )
    restored = priors.recipe_from_dict(json.loads(json.dumps(asdict(recipe))))
    assert restored == recipe
    task = priors.generate_task(restored, seed=28, task_id=28)
    expected = (2 * task.inputs[:, 0] - (4 if placement == "input" else 2)) / 1.0001
    torch.testing.assert_close(task.scores[:, 0], expected, rtol=1e-12, atol=1e-12)
    assert json.loads(task.mechanism_json)["moment_contract"]["target_orders"] == ["all"]
    assert priors.explain_generation(task, [0, 15])["target_max_error"] < 1e-12
    if isinstance(root, priors.RootInputCausalNode):
        with pytest.raises(ValueError, match="finite input second moments"):
            replace(
                recipe,
                nodes=(
                    replace(
                        root,
                        root_exogenous=priors.ExogenousDistribution(
                            inputs=priors.InputDistribution(distribution="student_t", df=1.5)
                        ),
                    ),
                    node,
                ),
            )


def test_standardization_keeps_representable_extreme_results_and_rejects_real_overflow() -> None:
    fitted = priors.calibrate_standardization(
        np.array([[0.0], [1.6e308]]), calibration_id="independent-large-finite-fit"
    )
    # The true centered numerator is -2.4e308, while the normalized value is -3.
    actual = fitted.apply(torch.tensor([[-1.6e308]], dtype=torch.float64))
    torch.testing.assert_close(
        actual, torch.tensor([[-3.0]], dtype=torch.float64), rtol=1e-14, atol=1e-14
    )
    one = priors.calibrate_standardization(np.array([[7.0]]), calibration_id="one-row-constant-fit")
    assert float(one.apply(torch.tensor([[8.0]], dtype=torch.float64))[0, 0]) == 10000.0
    with pytest.raises(ValueError, match="nonfinite"):
        one.apply(torch.tensor([[1e308]], dtype=torch.float64))


def test_standardization_admission_counts_frozen_constants_before_fitting() -> None:
    sample = np.array([[1.0, 7.0], [3.0, 7.0]])
    # Two means, two inverse scales and row count; inherited admission also
    # reserves two bias/noise slots per channel. 8*2*48 + 128*9 = 1920 bytes.
    exact = priors.CausalLimits(parameters=9, working_bytes=1920)
    fitted = priors.calibrate_standardization(
        sample, calibration_id="exact-fit-budget", limits=exact
    )
    assert fitted.parameter_count == 5
    for limits in (replace(exact, parameters=8), replace(exact, working_bytes=1919)):
        with pytest.raises(ValueError, match="limit"):
            priors.calibrate_standardization(
                sample, calibration_id="too-small-fit-budget", limits=limits
            )
