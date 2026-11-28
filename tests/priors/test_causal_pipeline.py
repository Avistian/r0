"""Node pipelines distinguish descendant state from explicitly converted readouts."""

import json
from dataclasses import asdict, replace

import numpy as np
import pytest
import torch

from r0.modules import priors


@pytest.mark.parametrize(
    ("variant", "feature", "target"),
    [
        ("historical-tuple", 0.5, 2.0),
        ("historical-graph", 0.25, 2.0),
        ("readout-only", 0.25, 4.0),
    ],
)
def test_converter_variants_separate_readout_and_descendant_state(
    variant: str, feature: float, target: float
) -> None:
    warp = priors.calibrate_kumaraswamy(
        np.array([[0.0], [1.0]]), a=2.0, b=1.0, calibration_id="independent-unit-range"
    )
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
            priors.CausalNodePipeline(
                channels=1,
                converters=(priors.NumericalConverter(channel=0, transform=warp, variant=variant),),
                node_importance=2.0,
            ),
            priors.CausalNodePipeline(channels=1),
        ),
        observed=((0, 0),),
        target=((1, 0),),
        version="causal-v2",
    )
    restored = priors.recipe_from_dict(json.loads(json.dumps(asdict(recipe))))
    assert restored == recipe
    task = priors.generate_task(restored, seed=28, task_id=4)
    torch.testing.assert_close(task.inputs, torch.full((16, 1), feature, dtype=torch.float64))
    torch.testing.assert_close(task.scores, torch.full((16, 1), target, dtype=torch.float64))
    assert priors.explain_generation(task, [0, 15])["target_max_error"] < 1e-12


@pytest.mark.parametrize("variant", ["historical-graph", "readout-only"])
def test_bounded_readout_moments_exclude_later_heavy_tailed_structural_noise(variant: str) -> None:
    warp = priors.calibrate_kumaraswamy(
        np.array([[-1.0], [1.0]]), a=1.0, b=1.0, calibration_id="independent-heavy-tail-range"
    )
    heavy = priors.ExogenousDistribution(
        inputs=priors.InputDistribution(distribution="student_t", df=0.5)
    )
    recipe = priors.CausalPipelineRecipe(
        task=priors.LinearRecipe(
            rows=16, context_rows=8, features=1, signal_features=1, active_features=1
        ),
        nodes=(
            priors.StochasticCausalNode(exogenous=heavy, noise_position="after_output"),
            priors.StochasticCausalNode(
                parents=(0,), weights=((1.0,),), exogenous=heavy, noise_position="after_output"
            ),
        ),
        pipelines=(
            priors.CausalNodePipeline(channels=1),
            priors.CausalNodePipeline(
                channels=1,
                converters=(priors.NumericalConverter(channel=0, transform=warp, variant=variant),),
            ),
        ),
        observed=((0, 0),),
        target=((1, 0),),
        version="causal-v2",
        normalize=True,
        calibration_rows=32,
    )
    task = priors.generate_task(recipe, seed=28, task_id=6)
    card = json.loads(task.mechanism_json)
    assert card["moment_contract"]["target_orders"] == ["all"]
    assert priors.explain_generation(task, [0, 15])["target_max_error"] < 1e-12


def test_readout_ancestry_does_not_include_later_structural_noise() -> None:
    recipe = priors.CausalPipelineRecipe(
        task=priors.LinearRecipe(
            rows=16, context_rows=8, features=1, signal_features=1, active_features=1
        ),
        nodes=(
            priors.ActivatedCausalNode(bias=(0.5,), noise_position="after_output"),
            priors.ActivatedCausalNode(
                parents=(0,), weights=((1.0,),), noise_scale=(0.0,), noise_position="after_output"
            ),
        ),
        pipelines=(
            priors.CausalNodePipeline(
                channels=1, converters=(priors.NumericalConverter(channel=0),)
            ),
            priors.CausalNodePipeline(channels=1),
        ),
        observed=((0, 0),),
        target=((1, 0),),
        version="causal-v2",
    )
    task = priors.generate_task(recipe, seed=28, task_id=11)
    torch.testing.assert_close(task.inputs, torch.full((16, 1), 0.5, dtype=torch.float64))
    assert float(task.scores.std()) > 0
    card = json.loads(task.mechanism_json)
    assert card["active"] == []
    assert "disconnected_target" in card["flags"]


def test_pipeline_orders_function_channel_importance_l2_converter_and_node_importance() -> None:
    l2 = priors.calibrate_l2_normalization(
        np.array([[3.0, 4.0], [-3.0, -4.0]]), calibration_id="independent-post-importance"
    )
    warp = priors.calibrate_kumaraswamy(
        np.array([[0.0], [1.0]]), a=2.0, b=1.0, calibration_id="independent-post-l2"
    )
    recipe = priors.CausalPipelineRecipe(
        task=priors.LinearRecipe(
            rows=16, context_rows=8, features=1, signal_features=1, active_features=1
        ),
        nodes=(
            priors.ActivatedCausalNode(
                bias=(2.0, 3.0), noise_scale=(0.0, 0.0), noise_position="after_output"
            ),
            priors.ActivatedCausalNode(
                parents=(0,),
                weights=((1.0,), (1.0,)),
                noise_scale=(0.0,),
                noise_position="after_output",
            ),
        ),
        pipelines=(
            priors.CausalNodePipeline(
                channels=2,
                stages=(priors.CausalActivation("square"),),
                channel_importance=(0.75, 4 / 9),
                l2=l2,
                converters=(priors.NumericalConverter(channel=0, transform=warp),),
                node_importance=2.0,
            ),
            priors.CausalNodePipeline(channels=1),
        ),
        observed=((0, 0),),
        target=((1, 0),),
        version="causal-v2",
    )
    restored = priors.recipe_from_dict(json.loads(json.dumps(asdict(recipe))))
    task = priors.generate_task(restored, seed=28, task_id=12)
    torch.testing.assert_close(
        task.inputs, torch.full((16, 1), 0.35999999856, dtype=torch.float64), rtol=1e-12, atol=1e-12
    )
    torch.testing.assert_close(
        task.scores, torch.full((16, 1), 2.7999999944, dtype=torch.float64), rtol=1e-12, atol=1e-12
    )
    trace = priors.explain_generation(task, [0, 15])
    assert trace["target_max_error"] < 1e-12
    nodes = {node["id"]: node for node in trace["nodes"]}
    assert nodes["inputs"]["parents"] == ["readout-0"]
    assert nodes["pipeline-0-stage-0"]["values"] == [[4.0, 9.0], [4.0, 9.0]]
    assert nodes["pipeline-0-channel-importance"]["values"] == [[3.0, 4.0], [3.0, 4.0]]
    assert nodes["pipeline-0-l2"]["values"][0] == pytest.approx([0.5999999988, 0.7999999984])


def test_l2_pipeline_requires_finite_moments_at_its_actual_input() -> None:
    l2 = priors.calibrate_l2_normalization(
        np.array([[-1.0], [1.0]]), calibration_id="independent-finite-sample"
    )
    warp = priors.calibrate_kumaraswamy(
        np.array([[-1.0], [1.0]]), a=1.0, b=1.0, calibration_id="independent-bounding-warp"
    )
    recipe = priors.CausalPipelineRecipe(
        task=priors.LinearRecipe(
            rows=16, context_rows=8, features=1, signal_features=1, active_features=1
        ),
        nodes=(
            priors.StochasticCausalNode(
                exogenous=priors.ExogenousDistribution(
                    inputs=priors.InputDistribution(distribution="student_t", df=0.5)
                ),
                noise_position="after_output",
            ),
            priors.ActivatedCausalNode(
                parents=(0,), weights=((1.0,),), noise_scale=(0.0,), noise_position="after_output"
            ),
        ),
        pipelines=(priors.CausalNodePipeline(channels=1), priors.CausalNodePipeline(channels=1)),
        observed=((0, 0),),
        target=((1, 0),),
        version="causal-v2",
    )
    with pytest.raises(ValueError, match="pipeline.*second moments"):
        replace(
            recipe, pipelines=(recipe.pipelines[0], priors.CausalNodePipeline(channels=1, l2=l2))
        )
    bounded = replace(
        recipe,
        pipelines=(
            recipe.pipelines[0],
            priors.CausalNodePipeline(channels=1, stages=(warp,), l2=l2),
        ),
    )
    task = priors.generate_task(bounded, seed=28, task_id=13)
    assert torch.isfinite(task.scores).all()
    assert json.loads(task.mechanism_json)["moment_contract"]["target_orders"] == ["all"]


def test_pipeline_admits_exact_geometry_and_rejects_malformed_saved_contracts() -> None:
    warp = priors.calibrate_kumaraswamy(
        np.array([[0.0], [1.0]]), a=2.0, b=1.0, calibration_id="independent-admission"
    )
    record = {
        "version": "causal-v2",
        "task": {
            "rows": 16,
            "context_rows": 8,
            "features": 1,
            "signal_features": 1,
            "active_features": 1,
        },
        "nodes": [
            {"bias": [0.5], "noise_scale": [0.0], "noise_position": "after_output"},
            {
                "parents": [0],
                "weights": [[4.0]],
                "noise_scale": [0.0],
                "noise_position": "after_output",
            },
        ],
        "observed": [[0, 0]],
        "target": [[1, 0]],
        "pipelines": [
            {
                "channels": 1,
                "node_importance": 2.0,
                "converters": [{"channel": 0, "transform": asdict(warp)}],
            },
            {"channels": 1},
        ],
        # Base equations: 5 numbers; pipelines: 2+8+2=12.
        # 8*16*(6*2+2*1+2*1+4*1+4+24+8) + 128*17 = 9344.
        "limits": {"parameters": 17, "working_bytes": 9344},
    }
    recipe = priors.recipe_from_dict(record)
    task = priors.generate_task(recipe, seed=28, task_id=14)
    admitted = json.loads(task.mechanism_json)["resource_admission"]
    assert admitted["parameters"] == 17
    assert admitted["estimated_working_bytes"] == 9344
    for field, value in (("parameters", 16), ("working_bytes", 9343)):
        altered = json.loads(json.dumps(record))
        altered["limits"][field] = value
        with pytest.raises(ValueError, match="limit"):
            priors.recipe_from_dict(altered)
    for field, value in (("channels", 2), ("version", "unsupported"), ("node_importance", 0)):
        altered = json.loads(json.dumps(record))
        altered["pipelines"][0][field] = value
        with pytest.raises(ValueError, match="width|version|importance"):
            priors.recipe_from_dict(altered)
    altered = json.loads(json.dumps(record))
    altered["pipelines"][0]["converters"] *= 2
    with pytest.raises(ValueError, match="unique"):
        priors.recipe_from_dict(altered)
    altered = json.loads(json.dumps(record))
    altered["nodes"][0]["noise_position"] = "before_output"
    with pytest.raises(ValueError, match="after-output"):
        priors.recipe_from_dict(altered)
