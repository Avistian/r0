"""Spatially varying smoothness and amplitude checked through generated tasks."""

import json
import math
from dataclasses import asdict, replace

import numpy as np
import pytest

from r0.modules.episodes import TaskType
from r0.modules.priors import (
    GaussianProcessRecipe,
    GibbsKernel,
    InputDistribution,
    LinearRecipe,
    StationaryKernel,
    generate_task,
    recipe_from_dict,
)


@pytest.mark.parametrize(
    ("task_type", "classes"), [("regression", 0), ("binary", 2), ("multiclass", 3)]
)
@pytest.mark.parametrize("amplitude_slopes", [(0.0, 0.0), (0.4, -0.2)])
def test_gibbs_draw_matches_variable_length_and_amplitude_covariance(
    amplitude_slopes: tuple[float, ...], task_type: TaskType, classes: int
) -> None:
    recipe = GaussianProcessRecipe(
        task=LinearRecipe(
            rows=16,
            context_rows=8,
            features=2,
            signal_features=2,
            active_features=2,
            task_type=task_type,
            classes=classes,
        ),
        calibration_rows=32,
        kernels=(
            GibbsKernel(
                length_scale=0.2,
                feature_scales=(0.7, 1.4),
                centers=(0.3, -0.5),
                transition_scales=(0.4, 1.2),
                length_slopes=(0.5, -0.8),
                amplitude_slopes=amplitude_slopes,
            ),
        ),
    )
    task = generate_task(recipe, seed=27, task_id=5)
    card = json.loads(task.mechanism_json)
    support = card["support"]
    covariance = np.empty((len(support), len(support)))
    for i, x in enumerate(support):
        for j, y in enumerate(support):
            value = 1.0
            for d, (base, center, transition, slope) in enumerate(
                ((0.14, 0.3, 0.4, 0.5), (0.28, -0.5, 1.2, -0.8))
            ):
                tx, ty = (
                    math.tanh((x[d] - center) / transition),
                    math.tanh((y[d] - center) / transition),
                )
                lx, ly = base * math.exp(slope * tx), base * math.exp(slope * ty)
                value *= math.sqrt(2 * lx * ly / (lx * lx + ly * ly)) * math.exp(
                    -((x[d] - y[d]) ** 2) / (lx * lx + ly * ly)
                )
                value *= math.exp(amplitude_slopes[d] * (tx + ty))
            covariance[i, j] = value
    eigenvalues, vectors = np.linalg.eigh(covariance)
    assert eigenvalues.min() > 0
    raw = ((vectors * np.sqrt(eigenvalues)) @ vectors.T) @ np.asarray(card["standard_normal"])
    calibration = raw[card["calibration_support_indices"]]
    expected = (raw[card["task_support_indices"]] - calibration.mean(axis=0)) / calibration.std(
        axis=0
    )
    np.testing.assert_allclose(task.scores.numpy(), expected, rtol=1e-10, atol=1e-10)
    decoded = recipe_from_dict(json.loads(json.dumps(asdict(recipe))))
    assert decoded == recipe
    assert generate_task(decoded, seed=27, task_id=5).episode() == task.episode()


def test_amplitude_product_overflow_has_an_explicit_failure() -> None:
    recipe = GaussianProcessRecipe(
        calibration_rows=32,
        combine="product",
        kernels=(GibbsKernel(centers=(-1e100,) * 4, amplitude_slopes=(3.0,) * 4),) * 32,
    )
    with pytest.raises(ValueError, match="combined kernel covariance"):
        generate_task(recipe, seed=27, task_id=0)


def test_constant_lengths_recover_rbf_and_paired_controls() -> None:
    recipe = GaussianProcessRecipe(calibration_rows=32, kernels=(GibbsKernel(length_scale=0.3),))
    original = generate_task(recipe, seed=27, task_id=2)
    baseline = generate_task(
        replace(recipe, kernels=(StationaryKernel(length_scale=0.3),)), seed=27, task_id=2
    )
    np.testing.assert_allclose(
        original.scores.numpy(), baseline.scores.numpy(), rtol=1e-10, atol=1e-10
    )
    for controls in (replace(recipe.task, features=12), replace(recipe.task, noise_ratio=2)):
        pair = generate_task(replace(recipe, task=controls), seed=27, task_id=2)
        assert original.mechanism_json == pair.mechanism_json
        np.testing.assert_array_equal(original.scores.numpy(), pair.scores.numpy())
        np.testing.assert_array_equal(
            original.context_indices.numpy(), pair.context_indices.numpy()
        )


def test_spatial_field_shapes_and_numerical_ranges_are_explicit() -> None:
    with pytest.raises(ValueError, match="match selected"):
        GaussianProcessRecipe(kernels=(GibbsKernel(centers=(0.0,)),))
    with pytest.raises(ValueError, match="transition"):
        GibbsKernel(transition_scales=(0,))
    with pytest.raises(ValueError, match="length slopes"):
        GibbsKernel(length_slopes=(100,))
    with pytest.raises(ValueError, match="amplitude slopes"):
        GibbsKernel(amplitude_slopes=(10, 10))
    with pytest.raises(ValueError, match="finite"):
        GibbsKernel(centers=(math.nan,))


def test_opposite_envelopes_cancel_regardless_of_product_order() -> None:
    base = GaussianProcessRecipe(
        task=LinearRecipe(
            rows=16, context_rows=8, features=1, signal_features=1, active_features=1
        ),
        inputs=InputDistribution(distribution="uniform", location=1.25, scale=0.25),
        calibration_rows=32,
        combine="product",
        kernels=(StationaryKernel(length_scale=0.001 / 8),),
    )
    expected = generate_task(base, seed=27, task_id=5)
    shrinking = GibbsKernel(length_scale=0.001, amplitude_slopes=(-13.0,))
    expanding = replace(shrinking, amplitude_slopes=(13.0,))
    for kernels in ((shrinking,) * 32 + (expanding,) * 32, (shrinking, expanding) * 32):
        actual = generate_task(replace(base, kernels=kernels), seed=27, task_id=5)
        np.testing.assert_allclose(
            actual.scores.numpy(), expected.scores.numpy(), rtol=1e-10, atol=1e-10
        )


def test_log_product_preserves_negative_spectral_covariance() -> None:
    from r0.modules.priors import SpectralKernel

    base = GaussianProcessRecipe(
        task=LinearRecipe(
            rows=16, context_rows=8, features=1, signal_features=1, active_features=1
        ),
        inputs=InputDistribution(distribution="uniform", location=1.25, scale=0.25),
        calibration_rows=32,
        combine="product",
        kernels=(SpectralKernel(length_scale=1 / math.sqrt(62 + 0.02**-2), frequencies=(7.0,)),),
    )
    expected = generate_task(base, seed=27, task_id=5)
    shrinking = GibbsKernel(amplitude_slopes=(-13.0,))
    expanding = replace(shrinking, amplitude_slopes=(13.0,))
    mixed = (
        (shrinking,) * 31
        + (SpectralKernel(length_scale=0.02, frequencies=(7.0,)),)
        + (expanding,) * 31
    )
    actual = generate_task(replace(base, kernels=mixed), seed=27, task_id=5)
    np.testing.assert_allclose(actual.scores.numpy(), expected.scores.numpy(), rtol=1e-9, atol=1e-9)
