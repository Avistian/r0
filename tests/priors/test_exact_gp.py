"""Direct GP draws checked through public recipes and covariance identities."""

import json
import math
from dataclasses import replace

import numpy as np
import pytest
import torch

from r0.modules.episodes import TaskType
from r0.modules.priors import (
    GaussianProcessRecipe,
    InputDistribution,
    LinearRecipe,
    StationaryKernel,
    generate_task,
    recipe_from_dict,
)


@pytest.mark.parametrize("kind", ["rbf", "matern12", "matern32", "matern52"])
def test_diagonal_covariance_has_known_draw_and_replays(kind: str) -> None:
    recipe = GaussianProcessRecipe(
        task=LinearRecipe(rows=16, context_rows=8),
        calibration_rows=32,
        inputs=InputDistribution(scale=1e100),
        kernels=(StationaryKernel(kind=kind),),
    )
    task = generate_task(recipe, seed=27, task_id=2)
    card = json.loads(task.mechanism_json)
    # All distinct supports are so far apart that these kernels round to identity.
    normals = np.asarray(card["standard_normal"], dtype=np.float64)
    raw = normals[np.asarray(card["task_support_indices"], dtype=np.int64)]
    expected = (raw - card["center"]) / card["scale"]
    np.testing.assert_allclose(task.scores.numpy(), expected, rtol=1e-10, atol=1e-10)
    decoded = recipe_from_dict(json.loads(task.recipe_json)["recipe"])
    assert task.episode() == generate_task(decoded, seed=27, task_id=2).episode()


@pytest.mark.parametrize("combine", ["sum", "product"])
@pytest.mark.parametrize(
    ("task_type", "classes"), [("regression", 0), ("binary", 2), ("multiclass", 3)]
)
def test_combined_subset_covariance_reconstructs_a_draw(
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
        calibration_rows=32,
        combine=combine,
        kernels=(
            StationaryKernel(
                length_scale=0.12,
                axes=(0,),
                feature_scales=(0.6,),
                weight=3 if combine == "sum" else 1,
            ),
            StationaryKernel(
                kind="matern52",
                length_scale=0.08,
                axes=(1,),
                feature_scales=(1.4,),
                weight=7 if combine == "sum" else 1,
            ),
        ),
    )
    task = generate_task(recipe, seed=27, task_id=5)
    card = json.loads(task.mechanism_json)
    support = card["support"]
    covariance = np.empty((len(support), len(support)))
    for i, x in enumerate(support):
        for j, y in enumerate(support):
            k0 = math.exp(-0.5 * ((x[0] - y[0]) / (0.12 * 0.6)) ** 2)
            t = math.sqrt(5) * abs(x[1] - y[1]) / (0.08 * 1.4)
            k1 = (1 + t + t * t / 3) * math.exp(-t)
            covariance[i, j] = 0.3 * k0 + 0.7 * k1 if combine == "sum" else k0 * k1
    eigenvalues, vectors = np.linalg.eigh(covariance)
    vectors *= np.sign(vectors[np.argmax(abs(vectors), axis=0), np.arange(len(support))])
    factor = (vectors * np.sqrt(np.maximum(eigenvalues, 0))) @ vectors.T
    np.testing.assert_allclose(factor @ factor.T, covariance, atol=card["roundoff_tolerance"])
    # The symmetric square root is independent of eigenvector signs/rotations.
    row = card["task_support_indices"][0]
    for c in range(task.scores.shape[1]):
        raw = math.fsum(
            float(a) * z[c] for a, z in zip(factor[row], card["standard_normal"], strict=True)
        )
        expected = (raw - card["center"][c]) / card["scale"][c]
        assert float(task.scores[0, c]) == pytest.approx(expected, abs=1e-10)


def test_gp_noise_nuisance_pairs_and_duplicate_inputs() -> None:
    recipe = GaussianProcessRecipe(calibration_rows=32)
    original = generate_task(recipe, seed=27, task_id=0)
    wide = generate_task(
        replace(recipe, task=replace(recipe.task, features=12)), seed=27, task_id=0
    )
    noisy = generate_task(
        replace(recipe, task=replace(recipe.task, noise_ratio=2)), seed=27, task_id=0
    )
    assert original.mechanism_json == wide.mechanism_json == noisy.mechanism_json
    for paired in (wide, noisy):
        torch.testing.assert_close(original.scores, paired.scores, rtol=0, atol=0)
        torch.testing.assert_close(original.context_indices, paired.context_indices, rtol=0, atol=0)
    duplicate = generate_task(
        GaussianProcessRecipe(
            task=LinearRecipe(
                rows=16, context_rows=8, features=1, signal_features=1, active_features=1
            ),
            calibration_rows=32,
            inputs=InputDistribution(location=1e16),
        ),
        seed=27,
        task_id=0,
    )
    observed: dict[float, float] = {}
    for x, score in zip(duplicate.inputs, duplicate.scores, strict=True):
        key, value = float(x[0]), float(score[0])
        if key in observed:
            assert value == observed[key]
        observed[key] = value
    assert len(observed) < 16


def test_reference_limits_and_unresolved_variation_fail_explicitly() -> None:
    with pytest.raises(ValueError, match="1024"):
        GaussianProcessRecipe(task=LinearRecipe(rows=20000))
    with pytest.raises(ValueError, match="cover every"):
        GaussianProcessRecipe(kernels=(StationaryKernel(axes=(0,)),))
    with pytest.raises(ValueError, match="roundoff resolution"):
        generate_task(
            GaussianProcessRecipe(kernels=(StationaryKernel(length_scale=1e200),)),
            seed=27,
            task_id=0,
        )


def test_weight_normalization_cannot_silently_remove_a_kernel() -> None:
    recipe = GaussianProcessRecipe(
        kernels=(StationaryKernel(weight=1e-300), StationaryKernel(weight=1e300)),
        calibration_rows=32,
    )
    with pytest.raises(ValueError, match="weight.*underflow"):
        generate_task(recipe, seed=27, task_id=0)


@pytest.mark.parametrize("kind", ["rbf", "matern12", "matern32", "matern52"])
def test_finite_scaled_distances_survive_raw_coordinate_overflow(kind: str) -> None:
    recipe = GaussianProcessRecipe(
        task=LinearRecipe(rows=16, context_rows=8, active_features=8),
        calibration_rows=32,
        inputs=InputDistribution(distribution="uniform", scale=8e307),
        kernels=(StationaryKernel(kind=kind, length_scale=1e308),),
    )
    task = generate_task(recipe, seed=27, task_id=0)
    card = json.loads(task.mechanism_json)
    x = np.asarray(card["support"], dtype=np.float64) / 1e308
    distance = np.linalg.norm(x[:, None, :] - x[None, :, :], axis=2)
    if kind == "rbf":
        covariance = np.exp(-(distance**2) / 2)
    else:
        t = distance * math.sqrt({"matern12": 1, "matern32": 3, "matern52": 5}[kind])
        covariance = np.exp(-t)
        if kind == "matern32":
            covariance *= 1 + t
        if kind == "matern52":
            covariance *= 1 + t + t * t / 3
    values, vectors = np.linalg.eigh(covariance)
    raw = ((vectors * np.sqrt(values)) @ vectors.T) @ np.asarray(card["standard_normal"])
    calibration = raw[np.asarray(card["calibration_support_indices"], dtype=np.int64)]
    expected = (
        raw[np.asarray(card["task_support_indices"], dtype=np.int64)] - calibration.mean(axis=0)
    ) / calibration.std(axis=0)
    np.testing.assert_allclose(task.scores.numpy(), expected, rtol=1e-10, atol=1e-10)
