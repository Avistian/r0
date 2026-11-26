"""Nonlinear structural equations through generation, replay and row inspection."""

import json
from dataclasses import asdict, replace

import pytest
import torch

from r0.modules.episodes import TaskType
from r0.modules.priors import (
    ActivatedCausalNode,
    CausalActivation,
    CausalNode,
    CausalRecipe,
    LinearRecipe,
    explain_generation,
    generate_task,
    recipe_from_dict,
    sample_recipe,
)


def test_nonlinear_noise_order_is_explicit_and_replayed_in_the_row_trace() -> None:
    controls = LinearRecipe(
        rows=16, context_rows=8, features=1, signal_features=1, active_features=1
    )
    node = ActivatedCausalNode(
        parents=(0,),
        weights=((2.0,),),
        bias=(1.0,),
        noise_scale=(0.5,),
        output_activation=CausalActivation("relu"),
    )
    before = CausalRecipe(
        task=controls,
        nodes=(CausalNode(bias=(-2.0,), noise_scale=(0.0,)), node),
        observed=((0, 0),),
        target=((1, 0),),
        version="causal-v2",
    )
    after = replace(before, nodes=(before.nodes[0], replace(node, noise_position="after_output")))
    left, right = [generate_task(r, seed=28, task_id=8) for r in (before, after)]
    traces = [explain_generation(t, list(range(16))) for t in (left, right)]
    nodes = [{n["id"]: n for n in trace["nodes"]} for trace in traces]
    assert nodes[0]["innovation-1"]["values"] == nodes[1]["innovation-1"]["values"]
    for row in range(16):
        epsilon = nodes[0]["innovation-1"]["values"][row][0]
        # Literal scalar equations: relu(-3 + epsilon) versus relu(-3) + epsilon.
        assert float(left.targets[row]) == max(0.0, -3.0 + epsilon)
        assert float(right.targets[row]) == epsilon
        assert nodes[0]["causal-1"]["values"][row][0] == float(left.targets[row])
        assert nodes[1]["causal-1"]["values"][row][0] == float(right.targets[row])
    assert not torch.equal(left.targets, right.targets)
    replay = generate_task(
        recipe_from_dict(json.loads(json.dumps(asdict(after)))), seed=28, task_id=8
    )
    assert replay.episode() == right.episode()
    assert "after_output" in nodes[1]["causal-1"]["parameters"]["noise_position"]


@pytest.mark.parametrize(
    ("kind", "classes", "noise"), [("regression", 0, 0.5), ("binary", 2, 0), ("multiclass", 3, 0)]
)
def test_activated_task_trace_reconstructs_target_adapter_draws(
    kind: TaskType, classes: int, noise: float
) -> None:
    controls = LinearRecipe(
        rows=32,
        context_rows=16,
        features=1,
        signal_features=1,
        active_features=1,
        task_type=kind,
        classes=classes,
        noise_ratio=noise,
    )
    width = classes if kind == "multiclass" else 1
    recipe = CausalRecipe(
        task=controls,
        nodes=(
            CausalNode(),
            ActivatedCausalNode(
                parents=(0,),
                weights=(tuple(float(i + 1) for i in range(width)),),
                bias=(0.0,) * width,
                noise_scale=(0.2,) * width,
                input_activation=CausalActivation("relu"),
            ),
        ),
        observed=((0, 0),),
        target=tuple((1, i) for i in range(width)),
        version="causal-v2",
    )
    task = generate_task(recipe, seed=28, task_id=9)
    trace = explain_generation(task, [0, 2, 5, 8, 10, 14, 16, 17, 23, 31])
    assert trace["target_max_error"] < 1e-12


def test_affine_v1_retains_its_pre_extension_task_identity() -> None:
    # Captured with the public API at 23f0302, before activated nodes were introduced.
    controls = LinearRecipe(rows=16, context_rows=8)
    recipe = sample_recipe("layered", profile="causal-affine-v1", task=controls, seed=28, task_id=7)
    task = generate_task(recipe, seed=28, task_id=7)
    assert task.source_task_id == "4a32001c1cc4b1245584567608a67dabb7d808894a4ca15224615380ad03adca"
    assert task.episode_id == "58796f5875753553826e6d780f42085319616ac02061e3b3942e841a2e373817"


@pytest.mark.parametrize(
    ("kind", "argument", "expected"),
    [
        ("identity", -2, -2),
        ("relu", -2, 0),
        ("tanh", -2, -0.9640275800758169),
        ("sigmoid", -2, 0.11920292202211755),
        ("leaky_relu", -2, -0.02),
        ("elu", -2, -0.8646647167633873),
        ("selu", 2, 2.101401974710961),
        ("silu", -2, -0.2384058440442351),
        ("softplus", 0, 0.6931471805599453),
        ("smooth_relu", 0, 0.6931471805599453),
        ("relu6", 9, 6),
        ("hardtanh", -2, -1),
        ("sign", 0, 0),
        ("sign_positive", 0, 1),
        ("gaussian_bump", -2, 0.01831563888873418),
        ("exp", 2, 7.38905609893065),
        ("sqrt_abs", -4, 2),
        ("interval_abs_le_1", -1, 1),
        ("interval_0_1", -1, 0),
        ("sin", 1.5707963267948966, 1),
        ("square", -2, 4),
        ("abs", -2, 2),
        ("log", 1, 0),
        ("log_abs", 0, -13.815510557964274),
        ("log_sigmoid", 0, -0.6931471805599453),
        ("round", 2.5, 2),
        ("step", 0, 0),
        ("heaviside", 0, 0),
        ("heaviside_positive", 0, 1),
        ("modulo", -1.25, 0.75),
        ("gelu", 1, 0.8413447460685429),
    ],
)
def test_fixed_activation_equations_survive_task_serialization(
    kind: str, argument: float, expected: float
) -> None:
    # Independent scalar reference values, including exact boundary conventions.
    controls = LinearRecipe(
        rows=16, context_rows=8, features=1, signal_features=1, active_features=1
    )
    recipe = CausalRecipe(
        task=controls,
        nodes=(
            ActivatedCausalNode(
                bias=(argument,), noise_scale=(0.0,), output_activation=CausalActivation(kind)
            ),
            CausalNode(),
        ),
        observed=((1, 0),),
        target=((0, 0),),
        version="causal-v2",
    )
    restored = recipe_from_dict(json.loads(json.dumps(asdict(recipe))))
    task = generate_task(restored, seed=28, task_id=10)
    torch.testing.assert_close(
        task.targets, torch.full((16,), expected, dtype=torch.float64), rtol=1e-12, atol=1e-12
    )
    trace = explain_generation(task, [0])
    node = next(n for n in trace["nodes"] if n["id"] == "causal-0")
    assert node["parameters"]["output_activation"]["kind"] == kind
    assert node["values"][0][0] == float(task.targets[0])


@pytest.mark.parametrize(
    ("activation", "argument", "expected"),
    [
        (
            CausalActivation(
                "identity",
                (
                    ("input_scale", -2),
                    ("input_shift", 1),
                    ("output_scale", 3),
                    ("output_shift", -4),
                ),
            ),
            2,
            -13,
        ),
        (CausalActivation("leaky_relu", (("negative_slope", 0.2),)), -2, -0.4),
        (CausalActivation("elu", (("alpha", 2),)), -2, -1.7293294335267746),
        (CausalActivation("softplus", (("beta", 2),)), 0, 0.34657359027997265),
        (CausalActivation("modulo", (("period", 2),)), -1.25, 0.75),
        (CausalActivation("log_abs", (("epsilon", 1),)), 0, 0),
    ],
)
def test_activation_parameters_change_the_frozen_function(
    activation: CausalActivation, argument: float, expected: float
) -> None:
    controls = LinearRecipe(
        rows=16, context_rows=8, features=1, signal_features=1, active_features=1
    )
    recipe = CausalRecipe(
        task=controls,
        nodes=(
            ActivatedCausalNode(bias=(argument,), noise_scale=(0.0,), output_activation=activation),
            CausalNode(),
        ),
        observed=((1, 0),),
        target=((0, 0),),
        version="causal-v2",
    )
    task = generate_task(
        recipe_from_dict(json.loads(json.dumps(asdict(recipe)))), seed=28, task_id=11
    )
    torch.testing.assert_close(
        task.targets, torch.full((16,), expected, dtype=torch.float64), rtol=1e-12, atol=1e-12
    )
