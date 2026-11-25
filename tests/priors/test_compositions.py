"""Shared-input compositions have independently checkable function arithmetic."""

import json
import math

import numpy as np
import pytest

from r0.modules.episodes import TaskType
from r0.modules.priors import (
    Combination,
    CompositeRecipe,
    FunctionComponent,
    LinearRecipe,
    PolynomialRecipe,
    SmoothRecipe,
    evaluate_signal,
    generate_task,
    recipe_from_dict,
)


@pytest.mark.parametrize(
    ("task_type", "classes"), [("regression", 0), ("binary", 2), ("multiclass", 3)]
)
def test_nested_shared_input_functions_match_saved_scalar_arithmetic(
    task_type: TaskType, classes: int
) -> None:
    controls = LinearRecipe(rows=16, context_rows=8, task_type=task_type, classes=classes)
    recipe = CompositeRecipe(
        task=controls,
        calibration_rows=256,
        components=(
            FunctionComponent(PolynomialRecipe(task=controls, degree=1, calibration_rows=256)),
            FunctionComponent(
                SmoothRecipe(task=controls, family="sinusoidal", terms=4, calibration_rows=256),
                strength=3,
            ),
        ),
        expression=Combination(
            "product", (Combination("sum", (0, 1), strength=2), 0), strength=-0.5
        ),
    )
    task = generate_task(recipe, seed=27, task_id=2)
    card = json.loads(task.mechanism_json)
    expected = np.empty(task.scores.shape)
    for i, row in enumerate(task.inputs.numpy()):
        for c in range(task.scores.shape[1]):
            leaves: list[float] = []
            for component in card["components"]:
                mechanism = component["mechanism"]
                x = [0.0] * component["feature_count"]
                for child_axis, global_axis in zip(
                    mechanism["active"], component["input_columns"], strict=True
                ):
                    x[child_axis] = float(row[global_axis])
                if mechanism["version"] == "polynomial-v1":
                    raw = math.fsum(
                        v**power * weight[c]
                        for v, power, weight in zip(
                            x, mechanism["powers"], mechanism["weights"], strict=True
                        )
                    )
                else:
                    raw = math.fsum(
                        math.sin(
                            (math.fsum(v * d for v, d in zip(x, direction, strict=True)) - center)
                            / width
                        )
                        * weight[c]
                        for direction, center, width, weight in zip(
                            mechanism["directions"],
                            mechanism["centers"],
                            mechanism["widths"],
                            mechanism["weights"],
                            strict=True,
                        )
                    )
                leaves.append((raw - mechanism["center"][c]) / mechanism["scale"][c])
            raw = -(leaves[0] + 3 * leaves[1]) * leaves[0]
            expected[i, c] = (raw - card["center"][c]) / card["scale"][c]
    np.testing.assert_allclose(task.scores.numpy(), expected, rtol=1e-10, atol=1e-10)
    assert set(card["active"]) == set(
        axis for comp in card["components"] for axis in comp["input_columns"]
    )
    assert card["interaction_order"] == 2
    replay = generate_task(
        recipe_from_dict(json.loads(task.recipe_json)["recipe"]), seed=27, task_id=2
    )
    assert replay.episode() == task.episode()
    np.testing.assert_array_equal(evaluate_signal(task, task.inputs).numpy(), task.scores.numpy())


def test_supported_component_count_and_depth_are_executable() -> None:
    controls = LinearRecipe(rows=16, context_rows=8)
    component = FunctionComponent(PolynomialRecipe(task=controls, degree=1, calibration_rows=256))
    many = CompositeRecipe(
        task=controls,
        components=(component,) * 64,
        expression=Combination("sum", tuple(range(64))),
        calibration_rows=256,
    )
    task = generate_task(many, seed=27, task_id=1)
    card = json.loads(task.mechanism_json)
    assert len({item["seed"] for item in card["components"]}) == 64
    expression: int | Combination = 0
    for _ in range(8):
        expression = Combination("product", (expression, 0))
    deep = generate_task(
        CompositeRecipe(
            task=controls, components=(component,), expression=expression, calibration_rows=256
        ),
        seed=27,
        task_id=1,
    )
    assert json.loads(deep.mechanism_json)["interaction_order"] == 9
    with pytest.raises(ValueError, match="depth"):
        CompositeRecipe(components=(component,), expression=Combination("sum", (expression, 0)))


def test_composed_moment_requirement_accounts_for_products() -> None:
    from r0.modules.priors import InputDistribution

    inputs = InputDistribution(distribution="student_t", df=5)
    component = FunctionComponent(PolynomialRecipe(inputs=inputs, degree=2, calibration_rows=256))
    with pytest.raises(ValueError, match="growth_degree"):
        CompositeRecipe(
            inputs=inputs, components=(component,), expression=Combination("product", (0, 0))
        )


def test_underflowing_component_strength_does_not_silently_zero_the_signal() -> None:
    controls = LinearRecipe(rows=16, context_rows=8)
    component = FunctionComponent(
        PolynomialRecipe(task=controls, degree=1, calibration_rows=256), strength=5e-324
    )
    recipe = CompositeRecipe(task=controls, components=(component,), calibration_rows=256)
    with pytest.raises(ValueError, match="composition.*underflow"):
        generate_task(recipe, seed=27, task_id=2)


def test_subset_axes_and_controlled_pairs_preserve_the_function() -> None:
    from dataclasses import replace

    import torch

    from r0.modules.priors import GroupedLinearRecipe

    controls = LinearRecipe(rows=16, context_rows=8)
    child = replace(controls, active_features=2)
    recipe = CompositeRecipe(
        task=controls,
        calibration_rows=256,
        components=(
            FunctionComponent(
                GroupedLinearRecipe(task=child, group_size=2, calibration_rows=256), axes=(0, 1)
            ),
            FunctionComponent(
                PolynomialRecipe(task=child, degree=2, calibration_rows=256), axes=(2, 3)
            ),
        ),
        expression=Combination("sum", (0, 1)),
    )
    original = generate_task(recipe, seed=27, task_id=4)
    for task_controls in (replace(controls, noise_ratio=2), replace(controls, features=12)):
        paired = generate_task(replace(recipe, task=task_controls), seed=27, task_id=4)
        assert paired.mechanism_json == original.mechanism_json
        torch.testing.assert_close(paired.scores, original.scores, rtol=0, atol=0)
        torch.testing.assert_close(paired.context_indices, original.context_indices, rtol=0, atol=0)
    inactive = set(range(controls.features)) - set(json.loads(original.mechanism_json)["active"])
    modified = original.inputs.clone()
    modified[:, list(inactive)] = 1e100
    torch.testing.assert_close(evaluate_signal(original, modified), original.scores, rtol=0, atol=0)
