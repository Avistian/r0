"""Bounded direct joint Gaussian draws, with declared covariance roundoff checks."""

from dataclasses import asdict, dataclass, field

import numpy as np
import torch

from r0.modules.priors.lib.calibration import stable_calibration
from r0.modules.priors.lib.inputs import InputDistribution, sample_inputs
from r0.modules.priors.lib.kernels import (
    Kernel,
    SpectralKernel,
    StationaryKernel,
    kernel_covariance,
)
from r0.modules.priors.lib.linear import LinearRecipe
from r0.modules.priors.lib.representatives import record_fingerprint
from r0.modules.priors.lib.task_output import complete_task
from r0.modules.priors.lib.training_task import TrainingTask


@dataclass(frozen=True, slots=True)
class GaussianProcessRecipe:
    task: LinearRecipe = field(default_factory=LinearRecipe)
    inputs: InputDistribution = field(default_factory=InputDistribution)
    kernels: tuple[Kernel, ...] = (StationaryKernel(),)
    combine: str = "sum"
    calibration_rows: int = 256
    version: str = "exact-gp-v1"

    def __post_init__(self) -> None:
        if self.version != "exact-gp-v1" or self.combine not in ("sum", "product"):
            raise ValueError("unsupported exact GP recipe")
        if self.task.rows > 1024:
            raise ValueError("exact GP reference tasks support at most 1024 rows")
        if type(self.calibration_rows) is not int or not 32 <= self.calibration_rows <= 512:
            raise ValueError("exact GP calibration rows must be 32..512")
        if type(self.kernels) is not tuple or not 1 <= len(self.kernels) <= 64:
            raise ValueError("kernels must be a tuple of 1..64 components")
        used: set[int] = set()
        for kernel in self.kernels:
            axes = kernel.axes or tuple(range(self.task.active_features))
            if any(v >= self.task.active_features for v in axes):
                raise ValueError("kernel axes index the selected active features")
            if kernel.feature_scales and len(kernel.feature_scales) != len(axes):
                raise ValueError("feature_scales must match selected kernel axes")
            if (
                isinstance(kernel, SpectralKernel)
                and kernel.frequencies
                and len(kernel.frequencies) != len(axes)
            ):
                raise ValueError("frequencies must match selected kernel axes")
            if self.combine == "product" and kernel.weight != 1:
                raise ValueError("product kernels require unit component weights")
            used.update(axes)
        if used != set(range(self.task.active_features)):
            raise ValueError("kernel components must cover every declared active feature")


def generate_exact_gp(
    recipe: GaussianProcessRecipe, *, seed: int, task_id: int, device: str = "cpu"
) -> TrainingTask:
    if any(type(v) is not int or not 0 <= v < 2**63 for v in (seed, task_id)):
        raise ValueError("seed and task_id must be integers in [0, 2**63)")
    if device not in ("cpu", "cuda") or (device == "cuda" and not torch.cuda.is_available()):
        raise ValueError("select an available CPU/CUDA device; no fallback")
    task = recipe.task

    def rng(*keys: object) -> np.random.Generator:
        return np.random.default_rng(
            int(record_fingerprint((recipe.version, seed, task_id, keys)), 16)
        )

    active = rng("active").permutation(task.signal_features)[: task.active_features]
    x = sample_inputs(recipe.inputs, task.rows, task.features, rng, "input")
    calibration = sample_inputs(
        recipe.inputs, recipe.calibration_rows, task.signal_features, rng, "calibration"
    )
    support, inverse = np.unique(
        np.vstack((calibration[:, active], x[:, active])), axis=0, return_inverse=True
    )
    cal_indices, task_indices = (
        inverse[: recipe.calibration_rows],
        inverse[recipe.calibration_rows :],
    )
    covariance, kernels = kernel_covariance(support, recipe.kernels, recipe.combine)
    eigenvalues, vectors = np.linalg.eigh(covariance)
    tolerance = 64 * np.finfo(np.float64).eps * len(support) * max(1.0, float(eigenvalues[-1]))
    if eigenvalues[0] < -tolerance:
        raise ValueError("covariance is indefinite beyond roundoff tolerance; task was not retried")
    expected_variance = float(
        np.diag(covariance)[cal_indices].mean()
        - covariance[np.ix_(cal_indices, cal_indices)].mean()
    )
    if expected_variance <= tolerance:
        raise ValueError(
            "calibration variation is below covariance roundoff resolution; task was not retried"
        )
    # The symmetric root removes arbitrary eigenvector signs and rotations.
    factor = (vectors * np.sqrt(np.maximum(eigenvalues, 0))) @ vectors.T
    covariance_error = float(np.max(np.abs(factor @ factor.T - covariance)))
    if not np.isfinite(factor).all() or covariance_error > tolerance:
        raise ValueError("GP factorization exceeds roundoff tolerance; task was not retried")
    outputs = task.classes if task.task_type == "multiclass" else 1
    normals = np.column_stack([rng("normal", c).normal(size=len(support)) for c in range(outputs)])
    raw_calibration = factor[cal_indices] @ normals
    center, scale = stable_calibration(raw_calibration)
    scores = torch.tensor(factor[task_indices], device=device) @ torch.tensor(
        normals, device=device
    )
    scores = (scores - torch.tensor(center, device=device)) / torch.tensor(scale, device=device)
    mechanism = {
        "family": "exact_gp",
        "version": recipe.version,
        "active": active.tolist(),
        "kernels": kernels,
        "combine": recipe.combine,
        "support": support.tolist(),
        "calibration_support_indices": cal_indices.tolist(),
        "task_support_indices": task_indices.tolist(),
        "standard_normal": normals.tolist(),
        "center": center.tolist(),
        "scale": scale.tolist(),
        "minimum_eigenvalue": float(eigenvalues[0]),
        "roundoff_tolerance": tolerance,
        "negative_eigenvalue_correction": max(0.0, -float(eigenvalues[0])),
        "factor_covariance_max_error": covariance_error,
        "expected_calibration_variance": expected_variance,
        "factorization": (
            "CPU numpy eigh symmetric square root; negative roundoff eigenvalues set to zero"
        ),
        "normalization": "estimated from independent calibration inputs in the same joint GP draw",
        "inputs": asdict(recipe.inputs),
        "calibration_rows": recipe.calibration_rows,
    }
    return complete_task(
        task,
        asdict(recipe),
        mechanism,
        torch.tensor(x, dtype=torch.float64, device=device),
        scores,
        seed=seed,
        task_id=task_id,
        rng=rng,
    )
