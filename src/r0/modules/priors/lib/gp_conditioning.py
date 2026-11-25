"""Bounded GP inspection, separate from the original task and its random streams."""

import json
import platform
from typing import Any

import numpy as np
import torch

from r0.modules.priors.lib.exact_gp import GaussianProcessRecipe
from r0.modules.priors.lib.generate import recipe_from_dict
from r0.modules.priors.lib.kernels import kernel_covariance
from r0.modules.priors.lib.representatives import record_fingerprint
from r0.modules.priors.lib.training_task import TrainingTask


def condition_gp(
    task: TrainingTask,
    inputs: torch.Tensor,
    *,
    draws: int = 0,
    seed: int | None = None,
    allow_rank_truncation: bool = False,
) -> dict[str, Any]:
    """Inspect noiseless latent values conditional on the complete saved GP draw.

    CPU inspection, 1..256 points and 0..8 additional draws with an explicit seed.
    Preserves saved values and normalization. No observation noise or target adapter.
    """
    recipe = recipe_from_dict(json.loads(task.recipe_json)["recipe"])
    if not isinstance(recipe, GaussianProcessRecipe):
        raise ValueError("conditional GP inspection requires an exact GP task")
    if (
        inputs.ndim != 2
        or not 1 <= len(inputs) <= 256
        or inputs.shape[1] != task.inputs.shape[1]
        or inputs.dtype != torch.float64
        or inputs.device.type != "cpu"
        or not torch.isfinite(inputs).all()
    ):
        raise ValueError("inputs must be finite float64 CPU rows, 1..256, matching task columns")
    if type(draws) is not int or not 0 <= draws <= 8:
        raise ValueError("draws must be an integer in 0..8")
    if (draws == 0 and seed is not None) or (
        draws > 0 and (type(seed) is not int or not 0 <= seed < 2**63)
    ):
        raise ValueError("provide a seed in [0, 2**63) only when requesting additional draws")
    if type(allow_rank_truncation) is not bool:
        raise ValueError("allow_rank_truncation must be a boolean")
    card = json.loads(task.mechanism_json)
    support = np.asarray(card["support"], dtype=np.float64)
    input_values = np.asarray(inputs.detach().numpy(), dtype=np.float64)
    points, inverse = np.unique(input_values[:, card["active"]], axis=0, return_inverse=True)
    points[points == 0] = 0  # Canonicalize signed zero in the query-set identity.
    joint, _ = kernel_covariance(np.vstack((support, points)), recipe.kernels, recipe.combine)
    n = len(support)
    covariance, cross = joint[:n, :n], joint[n:, :n]
    eigenvalues, vectors = np.linalg.eigh(covariance)
    tolerance = 64 * np.finfo(np.float64).eps * n * max(1.0, float(eigenvalues[-1]))
    if eigenvalues[0] < -tolerance:
        raise ValueError("saved GP covariance is indefinite beyond roundoff tolerance")
    retained = eigenvalues > tolerance
    rank = int(retained.sum())
    if rank < n and not allow_rank_truncation:
        raise ValueError(
            "GP support is numerically rank deficient; use allow_rank_truncation=True "
            "only for an explicitly approximate conditional view"
        )
    if rank == 0:
        raise ValueError("GP support has no numerically resolved modes")
    normals = np.asarray(card["standard_normal"], dtype=np.float64)
    raw = (vectors * np.sqrt(np.maximum(eigenvalues, 0))) @ vectors.T @ normals
    basis = vectors[:, retained]
    projected_normals = basis.T @ normals
    projected_support = (basis * np.sqrt(eigenvalues[retained])) @ projected_normals
    solved = (cross @ basis) / np.sqrt(eigenvalues[retained])
    mean = solved @ projected_normals
    conditional = joint[n:, n:] - solved @ solved.T
    conditional = (conditional + conditional.T) / 2
    # Known values are observations, not new random draws. Preserve them exactly.
    support_lookup = {tuple(row): i for i, row in enumerate(support)}
    matches = [support_lookup.get(tuple(row)) for row in points]
    for i, match in enumerate(matches):
        if match is not None:
            mean[i] = raw[match]
            conditional[i, :] = conditional[:, i] = 0
    conditional_eigenvalues, conditional_vectors = np.linalg.eigh(conditional)
    conditional_tolerance = (
        64 * np.finfo(np.float64).eps * (n + len(points)) * max(1.0, float(np.max(np.diag(joint))))
    )
    if conditional_eigenvalues[0] < -conditional_tolerance:
        raise ValueError("conditional covariance is indefinite beyond roundoff tolerance")
    root = (
        conditional_vectors * np.sqrt(np.maximum(conditional_eigenvalues, 0))
    ) @ conditional_vectors.T
    # Keep saved coordinates exactly fixed even after eigensolver roundoff.
    for i, match in enumerate(matches):
        if match is not None:
            root[i, :] = root[:, i] = 0
    conditional = root @ root.T
    center, scale = np.asarray(card["center"]), np.asarray(card["scale"])
    normalized_mean = (mean - center) / scale
    if not all(
        np.isfinite(value).all()
        for value in (normalized_mean, conditional, root / float(np.min(scale)))
    ):
        raise ValueError("nonfinite conditional values; inspection stopped without retry")
    query_fingerprint = record_fingerprint(points.tolist())
    samples: list[list[list[float]]] = []
    for draw in range(draws):
        key = ("gp-conditional-v1", task.source_task_id, query_fingerprint, seed, draw)
        rng = np.random.default_rng(int(record_fingerprint(key), 16))
        z = rng.normal(size=mean.shape)
        sample = (normalized_mean + (root @ z) / scale)[inverse]
        if not np.isfinite(sample).all():
            raise ValueError("nonfinite additional draw; inspection stopped without retry")
        samples.append(sample.tolist())
    return {
        "kind": "gp_conditional",
        "version": "gp-conditional-v1",
        "source_task_id": task.source_task_id,
        "episode_id": task.episode_id,
        "inputs": input_values.tolist(),
        "mean": normalized_mean[inverse].tolist(),
        "standard_deviation": (np.sqrt(np.diag(conditional))[:, None] / scale)[inverse].tolist(),
        "raw_covariance": conditional[np.ix_(inverse, inverse)].tolist(),
        "center": center.tolist(),
        "scale": scale.tolist(),
        "draws": samples,
        "original_support_indices": [matches[i] for i in inverse],
        "execution": {
            "device": "cpu",
            "python": platform.python_version(),
            "numpy": np.__version__,
            "torch": torch.__version__,
            "hardware": platform.machine(),
        },
        "sampling": {
            "seed": seed,
            "query_set_fingerprint": query_fingerprint,
            "method": "CPU numpy eigh symmetric square root of conditional covariance",
            "query_set_dependence": "a changed active query set changes additional draws",
        },
        "conditioning": {
            "support_count": n,
            "retained_rank": rank,
            "approximate": rank < n,
            "discarded_support_max_error": float(np.max(np.abs((raw - projected_support) / scale))),
            "rank_policy": "truncate" if allow_rank_truncation else "strict",
            "known_points": "return original values with zero uncertainty",
            "unknown_points": "condition on retained support eigenmodes",
            "observation_noise": 0,
            "eigenvalue_tolerance": tolerance,
            "conditional_roundoff_tolerance": conditional_tolerance,
            "negative_eigenvalue_correction": max(0.0, -float(conditional_eigenvalues[0])),
            "normalization": "original independent calibration center and scale",
        },
    }
