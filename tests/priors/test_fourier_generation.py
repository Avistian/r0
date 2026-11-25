"""Reconstruct nonlinear truth from the public task card, independently of Torch."""

import json
import math
from dataclasses import replace

import pytest
import torch

from r0.modules.priors import FourierRecipe, LinearRecipe, generate_fourier


@pytest.mark.parametrize("family", ["rff", "periodic"])
def test_fourier_card_reconstructs_a_row_and_replays(family: str) -> None:
    recipe = FourierRecipe(task=LinearRecipe(), terms=4, family=family)
    task = generate_fourier(recipe, seed=3, task_id=7)
    card = json.loads(task.mechanism_json)
    row = [float(task.inputs[0, j]) for j in range(recipe.task.signal_features)]
    terms = [
        math.cos(math.fsum(x * w for x, w in zip(row, weights, strict=True)) + phase)
        for weights, phase in zip(card["frequencies"], card["phases"], strict=True)
    ]
    raw = math.fsum(v * w[0] for v, w in zip(terms, card["coefficients"], strict=True))
    expected = (raw - card["center"][0]) / card["scale"][0]
    assert math.isclose(float(task.scores[0, 0]), expected, rel_tol=1e-12, abs_tol=1e-12)
    replay = generate_fourier(recipe, seed=3, task_id=7)
    assert task.episode() == replay.episode()


def test_fourier_pairs_preserve_mechanism_and_do_not_fit_query_rows() -> None:
    recipe = FourierRecipe(terms=8)
    first = generate_fourier(recipe, seed=11, task_id=2)
    wider = generate_fourier(
        replace(recipe, task=replace(recipe.task, features=12)), seed=11, task_id=2
    )
    taller = generate_fourier(
        replace(recipe, task=replace(recipe.task, rows=128)), seed=11, task_id=2
    )
    noisy = generate_fourier(
        replace(recipe, task=replace(recipe.task, noise_ratio=2)), seed=11, task_id=2
    )
    assert first.mechanism_json == wider.mechanism_json == taller.mechanism_json
    torch.testing.assert_close(first.scores, wider.scores, rtol=0, atol=0)
    torch.testing.assert_close(first.scores, noisy.scores, rtol=0, atol=0)
    torch.testing.assert_close(first.scores, taller.scores[:64], rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("classes", [2, 3, 100])
def test_classification_uses_shared_adapter_and_retains_weak_tasks(classes: int) -> None:
    task_type = "binary" if classes == 2 else "multiclass"
    task = generate_fourier(
        FourierRecipe(
            task=LinearRecipe(task_type=task_type, classes=classes, separation=0), terms=4
        ),
        seed=19,
        task_id=3,
    )
    assert task.probabilities is not None
    torch.testing.assert_close(task.probabilities, torch.full_like(task.probabilities, 1 / classes))
    assert len(task.episode().classes) == classes


@pytest.mark.parametrize("length_scale", [0, -1, float("nan"), float("inf")])
def test_invalid_length_scale_fails_before_sampling(length_scale: float) -> None:
    with pytest.raises(ValueError, match="length_scale"):
        FourierRecipe(length_scale=length_scale)


def test_periodic_components_cover_declared_active_features_and_share_a_period() -> None:
    with pytest.raises(ValueError, match="active feature"):
        FourierRecipe(family="periodic", terms=2)
    task = generate_fourier(FourierRecipe(family="periodic", terms=12), seed=7, task_id=0)
    card = json.loads(task.mechanism_json)
    used = {j for row in card["frequencies"] for j, value in enumerate(row) if value != 0}
    assert used == set(card["active"])
    for feature in used:
        frequencies = [row[feature] for row in card["frequencies"] if row[feature] != 0]
        period = 2 * math.pi / min(frequencies)
        for frequency in frequencies:
            assert math.cos(frequency * (0.37 + period)) == pytest.approx(
                math.cos(frequency * 0.37)
            )
