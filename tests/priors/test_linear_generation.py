"""Training generation is checked through its public interface."""

import math
from dataclasses import replace

import pytest
import torch

from r0.modules.priors import LinearRecipe, generate_linear


def test_single_feature_has_unit_signal_and_replays_after_other_tasks() -> None:
    recipe = LinearRecipe(features=1, signal_features=1, active_features=1)
    task = generate_linear(recipe, seed=19, task_id=7)
    # A normalized one-dimensional linear function is either x or -x.
    torch.testing.assert_close(task.scores[:, 0].abs(), task.inputs[:, 0].abs())
    generate_linear(recipe, seed=19, task_id=8)
    replay = generate_linear(recipe, seed=19, task_id=7)
    assert task.mechanism_json == replay.mechanism_json
    assert task.episode() == replay.episode()
    torch.testing.assert_close(task.targets, task.scores[:, 0])


def test_nuisance_columns_and_noise_preserve_active_inputs_and_function() -> None:
    recipe = LinearRecipe(features=4, signal_features=4, active_features=2)
    clean = generate_linear(recipe, seed=11, task_id=2)
    wider = generate_linear(replace(recipe, features=9), seed=11, task_id=2)
    noisy = generate_linear(replace(recipe, noise_ratio=2), seed=11, task_id=2)
    torch.testing.assert_close(clean.inputs, wider.inputs[:, :4], rtol=0, atol=0)
    torch.testing.assert_close(clean.scores, wider.scores, rtol=0, atol=0)
    torch.testing.assert_close(clean.scores, noisy.scores, rtol=0, atol=0)
    assert clean.mechanism_json == wider.mechanism_json == noisy.mechanism_json
    assert not torch.equal(clean.targets, noisy.targets)


def test_class_offsets_have_checkable_probabilities_without_forcing_counts() -> None:
    recipe = LinearRecipe(
        task_type="multiclass",
        classes=3,
        separation=0,
        class_offsets=(0.0, math.log(2), 0.0),
    )
    task = generate_linear(recipe, seed=23, task_id=0)
    expected = torch.tensor([[0.25, 0.5, 0.25]] * recipe.rows, dtype=torch.float64)
    assert task.probabilities is not None
    torch.testing.assert_close(task.probabilities, expected)
    view = task.episode().predictor_view()
    assert view.classes == ("class-0", "class-1", "class-2")
    assert not hasattr(view, "scores")
    assert not hasattr(view, "query_targets")


def test_binary_signal_matches_regression_and_declares_absent_classes() -> None:
    regression = LinearRecipe()
    binary = replace(regression, task_type="binary", classes=2, class_offsets=(1000.0,))
    first = generate_linear(regression, seed=31, task_id=9)
    second = generate_linear(binary, seed=31, task_id=9)
    torch.testing.assert_close(first.scores, second.scores, rtol=0, atol=0)
    assert second.episode().classes == ("class-0", "class-1")
    assert set(second.episode().context_targets) == {"class-1"}


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_device_is_explicit_and_gpu_agrees_with_cpu(device: str) -> None:
    recipe = LinearRecipe()
    if device == "cuda" and not torch.cuda.is_available():
        with pytest.raises(ValueError, match="CUDA is unavailable"):
            generate_linear(recipe, seed=13, task_id=2, device=device)
        pytest.skip("GPU numerical agreement NOT_RUN: CUDA unavailable")
    cpu = generate_linear(recipe, seed=13, task_id=2)
    result = generate_linear(recipe, seed=13, task_id=2, device=device)
    assert result.inputs.device.type == device
    torch.testing.assert_close(result.inputs.cpu(), cpu.inputs, rtol=0, atol=0)
    torch.testing.assert_close(result.scores.cpu(), cpu.scores, rtol=1e-10, atol=1e-12)


def test_split_changes_episode_but_not_source_and_predictor_view_is_a_snapshot() -> None:
    recipe = LinearRecipe()
    first = generate_linear(recipe, seed=7, task_id=3)
    second = generate_linear(replace(recipe, split=2, context_rows=20), seed=7, task_id=3)
    assert first.source_task_id == second.source_task_id
    assert first.episode_id != second.episode_id
    torch.testing.assert_close(first.inputs, second.inputs, rtol=0, atol=0)
    view = first.episode().predictor_view()
    saved = view.context[0].values
    first.inputs.fill_(999)
    assert view.context[0].values == saved


@pytest.mark.parametrize("noise", [float("nan"), float("inf"), -1, True])
def test_invalid_noise_is_rejected_before_generation(noise: float) -> None:
    with pytest.raises(ValueError, match="noise_ratio"):
        LinearRecipe(noise_ratio=noise)


def test_sparsity_controls_change_active_count_while_preserving_expected_signal_variance() -> None:
    import json

    for count in (1, 4, 8):
        task = generate_linear(LinearRecipe(active_features=count), seed=17, task_id=1)
        mechanism = json.loads(task.mechanism_json)
        weights = [row[0] for row in mechanism["weights"]]
        assert sum(w != 0 for w in weights) == count
        # Independent Gaussian columns: Var(sum w_j X_j) = sum w_j^2.
        assert math.fsum(w * w for w in weights) == pytest.approx(1.0)
