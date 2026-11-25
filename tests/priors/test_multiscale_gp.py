"""Public GP tasks checked against independently constructed covariance matrices."""

import json
import math
from dataclasses import asdict, replace
from typing import Any

import numpy as np
import pytest
from numpy.typing import NDArray

from r0.modules.episodes import TaskType
from r0.modules.priors import (
    GaussianProcessRecipe,
    LinearRecipe,
    RationalQuadraticKernel,
    SpectralKernel,
    StationaryKernel,
    generate_task,
    recipe_from_dict,
)


def test_rational_quadratic_draw_matches_multiscale_covariance() -> None:
    recipe = GaussianProcessRecipe(
        task=LinearRecipe(
            rows=16, context_rows=8, features=2, signal_features=2, active_features=2
        ),
        kernels=(RationalQuadraticKernel(length_scale=0.2, feature_scales=(0.7, 1.4), alpha=0.3),),
        calibration_rows=32,
    )
    task = generate_task(recipe, seed=27, task_id=5)
    card = json.loads(task.mechanism_json)
    support = card["support"]
    covariance = np.array(
        [
            [
                (
                    1
                    + math.fsum(
                        ((x[d] - y[d]) / length) ** 2 for d, length in enumerate((0.14, 0.28))
                    )
                    / 0.6
                )
                ** -0.3
                for y in support
            ]
            for x in support
        ]
    )
    _check_draw(task.scores.numpy(), card, covariance)
    assert recipe_from_dict(json.loads(json.dumps(asdict(recipe)))) == recipe
    assert (
        generate_task(recipe_from_dict(asdict(recipe)), seed=27, task_id=5).episode()
        == task.episode()
    )


def _check_draw(
    scores: NDArray[np.float64], card: dict[str, Any], covariance: NDArray[np.float64]
) -> None:
    values, vectors = np.linalg.eigh(covariance)
    assert values.min() > 0  # These examples are well conditioned; no clipping in the oracle.
    raw = ((vectors * np.sqrt(values)) @ vectors.T) @ np.asarray(card["standard_normal"])
    calibration = raw[card["calibration_support_indices"]]
    expected = (raw[card["task_support_indices"]] - calibration.mean(axis=0)) / calibration.std(
        axis=0
    )
    np.testing.assert_allclose(scores, expected, rtol=1e-10, atol=1e-10)


@pytest.mark.parametrize("combine", ["sum", "product"])
@pytest.mark.parametrize(
    ("task_type", "classes"), [("regression", 0), ("binary", 2), ("multiclass", 3)]
)
def test_frequency_bands_match_published_product_of_cosines(
    combine: str, task_type: TaskType, classes: int
) -> None:
    recipe = GaussianProcessRecipe(
        task=LinearRecipe(
            rows=16,
            context_rows=8,
            features=3,
            signal_features=2,
            active_features=2,
            task_type=task_type,
            classes=classes,
        ),
        kernels=(
            SpectralKernel(
                length_scale=0.25,
                frequencies=(0.7, 1.2),
                feature_scales=(0.8, 1.4),
                weight=2 if combine == "sum" else 1,
            ),
            SpectralKernel(
                length_scale=0.5, frequencies=(0.3, 2.5), weight=3 if combine == "sum" else 1
            ),
        ),
        combine=combine,
        calibration_rows=32,
    )
    task = generate_task(recipe, seed=27, task_id=5)
    card = json.loads(task.mechanism_json)
    support = card["support"]
    covariance = np.empty((len(support), len(support)))
    for i, x in enumerate(support):
        for j, y in enumerate(support):
            first = math.prod(
                math.exp(-0.5 * ((x[d] - y[d]) / length) ** 2)
                * math.cos(2 * math.pi * (x[d] - y[d]) * frequency)
                for d, (length, frequency) in enumerate(((0.2, 0.7), (0.35, 1.2)))
            )
            second = math.prod(
                math.exp(-0.5 * ((x[d] - y[d]) / 0.5) ** 2)
                * math.cos(2 * math.pi * (x[d] - y[d]) * frequency)
                for d, frequency in enumerate((0.3, 2.5))
            )
            covariance[i, j] = 0.4 * first + 0.6 * second if combine == "sum" else first * second
    assert covariance.min() < 0
    _check_draw(task.scores.numpy(), card, covariance)
    assert recipe_from_dict(json.loads(json.dumps(asdict(recipe)))) == recipe
    assert (
        generate_task(recipe_from_dict(asdict(recipe)), seed=27, task_id=5).episode()
        == task.episode()
    )


def test_large_alpha_and_zero_frequency_recover_rbf_draw() -> None:
    recipe = GaussianProcessRecipe(
        calibration_rows=32, kernels=(StationaryKernel(length_scale=0.3),)
    )
    baseline = generate_task(recipe, seed=27, task_id=0)
    for kernel in (
        RationalQuadraticKernel(length_scale=0.3, alpha=1e308),
        SpectralKernel(length_scale=0.3),
    ):
        task = generate_task(replace(recipe, kernels=(kernel,)), seed=27, task_id=0)
        np.testing.assert_allclose(
            task.scores.numpy(), baseline.scores.numpy(), rtol=1e-10, atol=1e-10
        )


def test_unresolved_spectral_phase_fails_instead_of_changing_frequency() -> None:
    with pytest.raises(ValueError, match="phase exceeds"):
        generate_task(
            GaussianProcessRecipe(
                kernels=(SpectralKernel(frequencies=(1e100,) * 4),), calibration_rows=32
            ),
            seed=27,
            task_id=0,
        )


@pytest.mark.parametrize("alpha", [0, -1, math.inf, math.nan, True])
def test_invalid_multiscale_shape_is_rejected(alpha: float) -> None:
    with pytest.raises(ValueError, match="alpha"):
        RationalQuadraticKernel(alpha=alpha)


def test_frequency_axes_and_units_are_checked() -> None:
    with pytest.raises(ValueError, match="frequencies.*axes"):
        GaussianProcessRecipe(kernels=(SpectralKernel(frequencies=(1.0,)),))
    for frequencies in ((-1.0,), (math.inf,), (True,)):
        with pytest.raises(ValueError, match="frequencies"):
            SpectralKernel(frequencies=frequencies)
