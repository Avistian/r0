"""Controlled changes preserve the declared shared task properties."""

import json

import numpy as np
import torch

from r0.modules.priors import LinearRecipe, generate_pair


def test_noise_pair_preserves_mechanism_and_scales_the_same_residuals() -> None:
    pair = generate_pair(
        LinearRecipe(noise_ratio=0.2),
        intervention="noise",
        value=0.8,
        seed=27,
        task_id=9,
    )
    left, right = pair.before, pair.after
    assert left.mechanism_json == right.mechanism_json
    assert torch.equal(left.inputs, right.inputs)
    assert torch.equal(left.scores, right.scores)
    assert torch.equal(left.context_indices, right.context_indices)
    np.testing.assert_allclose(
        right.targets.numpy() - right.scores[:, 0].numpy(),
        4 * (left.targets.numpy() - left.scores[:, 0].numpy()),
        atol=1e-15,
    )
    card = json.loads(pair.card_json)
    assert card["intervention"] == "noise"
    assert card["before"]["source_task_id"] == left.source_task_id
    assert card["after"]["source_task_id"] == right.source_task_id
    assert all(card["required_equalities"].values())
    assert card["independent_source_draws"] == 1
    assert np.isclose(card["after"]["residual_std"], 4 * card["before"]["residual_std"])
    replay = generate_pair(
        LinearRecipe(noise_ratio=0.2), intervention="noise", value=0.8, seed=27, task_id=9
    )
    assert replay.card_json == pair.card_json
    assert replay.before.episode() == left.episode()
    assert replay.after.episode() == right.episode()


def test_irrelevant_columns_and_class_separation_have_distinct_fixed_properties() -> None:
    from r0.modules.priors import FourierRecipe

    recipe = FourierRecipe(task=LinearRecipe(task_type="multiclass", classes=3), terms=64)
    nuisance = generate_pair(recipe, intervention="irrelevant", value=10, seed=27, task_id=3)
    assert torch.equal(nuisance.before.inputs, nuisance.after.inputs[:, :8])
    assert torch.equal(nuisance.before.targets, nuisance.after.targets)
    assert nuisance.before.probabilities is not None
    assert nuisance.after.probabilities is not None
    assert torch.equal(nuisance.before.probabilities, nuisance.after.probabilities)
    assert nuisance.before.mechanism_json == nuisance.after.mechanism_json
    separated = generate_pair(recipe, intervention="separation", value=0, seed=27, task_id=3)
    assert torch.equal(separated.before.scores, separated.after.scores)
    assert separated.before.mechanism_json == separated.after.mechanism_json
    assert separated.after.probabilities is not None
    np.testing.assert_allclose(separated.after.probabilities.numpy(), 1 / 3)
    report = json.loads(separated.card_json)
    assert np.isclose(report["after"]["mean_entropy"], np.log(3), atol=1e-14)
    assert sum(report["after"]["class_counts"]) == 64


def test_sparse_linear_pair_keeps_expected_variance_not_realized_sample_variance() -> None:
    pair = generate_pair(LinearRecipe(), intervention="sparsity", value=2, seed=27, task_id=5)
    before, after = (json.loads(task.mechanism_json) for task in (pair.before, pair.after))
    assert set(after["active"]) < set(before["active"])
    np.testing.assert_allclose(np.sum(np.array(before["weights"]) ** 2, axis=0), 1)
    np.testing.assert_allclose(np.sum(np.array(after["weights"]) ** 2, axis=0), 1)
    assert torch.equal(pair.before.inputs, pair.after.inputs)
    assert not torch.equal(pair.before.scores, pair.after.scores)
    report = json.loads(pair.card_json)
    assert report["variance_control"] == "analytic_unit_variance"
    assert report["observed_equalities"]["scores"] is False
    assert "scores" not in report["required_equalities"]
    assert all(report["required_equalities"].values())


def test_composition_sparsity_preserves_expression_and_fixed_component_mapping() -> None:
    from r0.modules.priors import Combination, CompositeRecipe, FunctionComponent, PolynomialRecipe

    recipe = CompositeRecipe(
        components=(FunctionComponent(LinearRecipe()), FunctionComponent(PolynomialRecipe())),
        expression=Combination("product", (0, 1)),
        calibration_rows=256,
    )
    import pytest

    with pytest.raises(ValueError, match="expected-variance"):
        generate_pair(recipe, intervention="sparsity", value=2, seed=27, task_id=8)
    pair = generate_pair(
        recipe, intervention="sparsity", value=2, seed=27, task_id=8, allow_estimated_variance=True
    )
    before, after = (json.loads(task.mechanism_json) for task in (pair.before, pair.after))
    assert before["expression"] == after["expression"]
    assert [c["seed"] for c in before["components"]] == [c["seed"] for c in after["components"]]
    assert all(len(c["input_columns"]) == 2 for c in after["components"])
    report = json.loads(pair.card_json)
    assert report["variance_control"] == "independent_calibration_unit_variance_estimate"
    assert report["after"]["expected_signal_variance"] is None


def test_pair_rejects_extra_changes_different_draws_and_invalid_geometry() -> None:
    from dataclasses import replace

    import pytest

    from r0.modules.priors import GroupedLinearRecipe, compare_pair, generate_task, paired_recipe

    before = generate_task(LinearRecipe(noise_ratio=0.1), seed=27, task_id=2)
    changed = generate_task(LinearRecipe(noise_ratio=0.5, active_features=2), seed=27, task_id=2)
    with pytest.raises(ValueError, match="outside"):
        compare_pair(before, changed, intervention="noise")
    other_draw = generate_task(LinearRecipe(noise_ratio=0.5), seed=28, task_id=2)
    with pytest.raises(ValueError, match="required equalities"):
        compare_pair(before, other_draw, intervention="noise")
    right = generate_task(LinearRecipe(noise_ratio=0.5), seed=27, task_id=2)
    corrupted = replace(right, scores=right.scores + 1)
    with pytest.raises(ValueError, match="scores"):
        compare_pair(before, corrupted, intervention="noise")
    with pytest.raises(ValueError, match="whole multiples"):
        paired_recipe(GroupedLinearRecipe(), intervention="sparsity", value=3)
    with pytest.raises(ValueError, match="must change"):
        paired_recipe(LinearRecipe(), intervention="noise", value=0)
    with pytest.raises(ValueError, match="classification"):
        paired_recipe(LinearRecipe(task_type="binary", classes=2), intervention="noise", value=0.1)
    with pytest.raises(ValueError, match="unsupported"):
        paired_recipe(LinearRecipe(), intervention="other", value=2)
