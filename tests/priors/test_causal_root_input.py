"""Root mechanism inputs remain separate from later structural innovations."""

import json
from dataclasses import asdict, replace

import pytest
import torch

from r0.modules import priors


def test_root_draw_survives_pipeline_conversion_before_independent_innovation() -> None:
    root = priors.RootInputCausalNode(
        bias=(0.5, 0.75),
        root_scale=(2.0, -1.0),
        noise_scale=(0.25, 0.5),
        output_activation=priors.CausalActivation("tanh"),
    )
    recipe = priors.CausalPipelineRecipe(
        task=priors.LinearRecipe(
            rows=16, context_rows=8, features=2, signal_features=2, active_features=2
        ),
        nodes=(
            root,
            priors.ActivatedCausalNode(
                parents=(0,),
                weights=((3.0,), (-1.0,)),
                noise_scale=(0.0,),
                noise_position="after_output",
            ),
        ),
        pipelines=(
            priors.CausalNodePipeline(
                channels=2, converters=(priors.NumericalConverter(channel=0),), node_importance=2.0
            ),
            priors.CausalNodePipeline(channels=1),
        ),
        observed=((0, 0), (0, 1)),
        target=((1, 0),),
        version="causal-v2",
    )
    restored = priors.recipe_from_dict(json.loads(json.dumps(asdict(recipe))))
    assert restored == recipe
    task = priors.generate_task(restored, seed=28, task_id=20)
    quiet = priors.generate_task(
        replace(recipe, nodes=(replace(root, noise_scale=(0.0, 0.0)), recipe.nodes[1])),
        seed=28,
        task_id=20,
    )
    torch.testing.assert_close(task.inputs[:, 0], quiet.inputs[:, 0], rtol=0, atol=0)
    trace = priors.explain_generation(task, list(range(16)))
    nodes = {node["id"]: node for node in trace["nodes"]}
    assert tuple(nodes["noise-scale-0"]["parameters"]["root_scale"]) == (2.0, -1.0)
    raw = torch.tensor(nodes["root-input-0"]["values"], dtype=torch.float64)
    exogenous = torch.tensor(nodes["exogenous-0"]["values"], dtype=torch.float64)
    assert not torch.equal(raw, exogenous)
    assert torch.std(task.inputs[:, 0]) > 0
    expected_first = torch.tanh(0.5 + 2 * raw[:, 0])
    torch.testing.assert_close(task.inputs[:, 0], expected_first, rtol=1e-12, atol=1e-12)
    innovation = torch.tensor(nodes["innovation-0"]["values"], dtype=torch.float64)
    torch.testing.assert_close(
        task.scores[:, 0] - quiet.scores[:, 0],
        2 * (3 * innovation[:, 0] - innovation[:, 1]),
        rtol=1e-12,
        atol=1e-12,
    )
    assert trace["target_max_error"] < 1e-12
    assert json.loads(task.mechanism_json)["active"] == [0, 1]


@pytest.mark.parametrize(
    ("df", "strength", "power", "order"), [(0.5, 0.0, 1.0, 0.5), (3.0, 1.0, 2.0, 1.5)]
)
def test_root_moments_and_signal_dependent_innovation_follow_the_actual_affine_input(
    df: float, strength: float, power: float, order: float
) -> None:
    root = priors.RootInputCausalNode(
        bias=(0.0, 0.0),
        root_scale=(1.0, 1.0),
        noise_scale=(1.0, 1.0),
        root_exogenous=priors.ExogenousDistribution(
            inputs=priors.InputDistribution(distribution="student_t", df=df),
            factor=((1.0, 2.0), (0.0, 0.0)),
        ),
        noise_response=priors.NoiseResponse(offset=0.0, strength=strength, power=power),
    )
    recipe = priors.CausalRecipe(
        task=priors.LinearRecipe(
            rows=16, context_rows=8, features=1, signal_features=1, active_features=1
        ),
        nodes=(root,),
        observed=((0, 0),),
        target=((0, 1),),
        version="causal-v2",
    )
    task = priors.generate_task(recipe, seed=28, task_id=21)
    card = json.loads(task.mechanism_json)
    assert card["moment_contract"]["target_orders"] == [order]
    assert card["active"] == [0]
    with pytest.raises(ValueError, match="second moments"):
        replace(recipe, normalize=True)


@pytest.mark.parametrize(
    "law",
    [
        priors.ExogenousDistribution(),
        priors.ExogenousDistribution(inputs=priors.InputDistribution(distribution="uniform")),
        priors.ExogenousDistribution(
            inputs=priors.InputDistribution(distribution="student_t", df=3.0)
        ),
        priors.ExogenousDistribution(inputs=priors.InputDistribution(distribution="mixture")),
        priors.ExogenousDistribution(
            inputs=priors.InputDistribution(distribution="student_t", df=0.5, bound=2.0)
        ),
        priors.ExogenousDistribution(geometry="ball"),
        priors.ExogenousDistribution(factor=((1.0, 2.0), (0.0, 0.0))),
    ],
)
def test_root_laws_replay_original_rows_and_preserve_context_prefix(
    law: priors.ExogenousDistribution,
) -> None:
    recipe = priors.CausalRecipe(
        task=priors.LinearRecipe(
            rows=16, context_rows=8, features=1, signal_features=1, active_features=1
        ),
        nodes=(
            priors.RootInputCausalNode(
                root_exogenous=law,
                bias=(0.25, -0.5),
                root_scale=(-2.0, 0.5),
                noise_scale=(0.3, 0.7),
                noise_position="before_output",
                output_activation=priors.CausalActivation("tanh"),
            ),
        ),
        observed=((0, 0),),
        target=((0, 1),),
        version="causal-v2",
    )
    task = priors.generate_task(recipe, seed=28, task_id=25)
    longer = priors.generate_task(
        replace(recipe, task=replace(recipe.task, rows=32, context_rows=12)), seed=28, task_id=25
    )
    torch.testing.assert_close(task.inputs, longer.inputs[:16], rtol=0, atol=0)
    torch.testing.assert_close(task.scores, longer.scores[:16], rtol=0, atol=0)
    trace = priors.explain_generation(task, [15, 0, 8])
    steps = {node["id"]: node for node in trace["nodes"]}
    root = torch.tensor(steps["root-input-0"]["values"], dtype=torch.float64)
    innovation = torch.tensor(steps["innovation-0"]["values"], dtype=torch.float64)
    expected = torch.tanh(-0.5 + 0.5 * root[:, 1] + innovation[:, 1])
    torch.testing.assert_close(task.scores[[15, 0, 8], 0], expected, rtol=1e-12, atol=1e-12)
    assert trace["target_max_error"] < 1e-12
