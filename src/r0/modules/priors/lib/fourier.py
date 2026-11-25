"""Finite trigonometric functions with independent population calibration."""

import math
from dataclasses import asdict, dataclass, field

import numpy as np
import torch
from numpy.typing import NDArray

from r0.modules.priors.lib.calibration import stable_calibration
from r0.modules.priors.lib.linear import LinearRecipe
from r0.modules.priors.lib.representatives import record_fingerprint
from r0.modules.priors.lib.signals import evaluate_mechanism
from r0.modules.priors.lib.task_output import complete_task
from r0.modules.priors.lib.training_task import TrainingTask


@dataclass(frozen=True, slots=True)
class FourierRecipe:
    task: LinearRecipe = field(default_factory=LinearRecipe)
    family: str = "rff"
    terms: int = 256
    length_scale: float = 1.0
    calibration_rows: int = 4096
    version: str = "fourier-v1"

    def __post_init__(self) -> None:
        if self.version != "fourier-v1" or self.family not in ("rff", "periodic"):
            raise ValueError("unsupported Fourier recipe")
        if type(self.terms) is not int or not 1 <= self.terms <= 4096:
            raise ValueError("terms must be 1..4096")
        if self.family == "periodic" and self.terms > 64:
            raise ValueError("periodic mixtures support at most 64 components")
        if self.family == "periodic" and self.terms < self.task.active_features:
            raise ValueError("periodic mixtures need at least one component per active feature")
        if type(self.length_scale) not in (int, float) or not (
            math.isfinite(self.length_scale) and self.length_scale > 0
        ):
            raise ValueError("length_scale must be finite and positive")
        if type(self.calibration_rows) is not int or not 256 <= self.calibration_rows <= 16384:
            raise ValueError("calibration_rows must be 256..16384")


def generate_fourier(
    recipe: FourierRecipe, *, seed: int, task_id: int, device: str = "cpu"
) -> TrainingTask:
    """Draw a finite RFF function or a mixture of single-feature periodic functions.

    Calibration is independent of the task rows. Unit variance is estimated, not an
    exact population guarantee. RFF is explicitly a finite approximation, not an exact GP.
    """
    if any(type(v) is not int or not 0 <= v < 2**63 for v in (seed, task_id)):
        raise ValueError("seed and task_id must be integers in [0, 2**63)")
    if device not in ("cpu", "cuda") or (device == "cuda" and not torch.cuda.is_available()):
        raise ValueError("select an available CPU/CUDA device; no fallback")
    task = recipe.task

    def rng(*keys: object) -> np.random.Generator:
        return np.random.default_rng(
            int(record_fingerprint((recipe.version, recipe.family, seed, task_id, keys)), 16)
        )

    active = rng("active").permutation(task.signal_features)[: task.active_features]
    frequencies = np.stack(
        [
            rng("frequency", j).normal(size=task.signal_features) / recipe.length_scale
            for j in range(recipe.terms)
        ]
    )
    frequencies[:, np.setdiff1d(np.arange(task.signal_features), active)] = 0
    if recipe.family == "periodic":
        base = rng("base_frequency").uniform(0.25, 4, size=task.signal_features)
        frequencies.fill(0)
        for j in range(recipe.terms):
            chosen = active[j % len(active)]
            frequencies[j, chosen] = base[chosen] * (1 + j // len(active)) / recipe.length_scale
    phases = rng("phase").uniform(-math.pi, math.pi, size=recipe.terms)
    outputs = task.classes if task.task_type == "multiclass" else 1
    coefficients = np.column_stack(
        [
            rng("coefficient", j).normal(size=recipe.terms) / math.sqrt(recipe.terms)
            for j in range(outputs)
        ]
    )
    # CPU calibration makes the stored mechanism identical across execution devices.
    calibration = np.column_stack(
        [
            rng("calibration", j).normal(size=recipe.calibration_rows)
            for j in range(task.signal_features)
        ]
    )

    def calibration_samples(*, centered: bool) -> NDArray[np.float64]:
        parts: list[NDArray[np.float64]] = []
        for part in np.array_split(calibration, math.ceil(recipe.calibration_rows / 256)):
            projection = part @ frequencies.T
            basis = (
                -2 * np.sin(projection / 2) * np.sin(phases + projection / 2)
                if centered
                else np.cos(projection + phases)
            )
            parts.append(basis @ coefficients)
        return np.concatenate(parts)

    samples = calibration_samples(centered=False)
    center, scale = samples.mean(axis=0), samples.std(axis=0)
    centered_basis = bool(np.any(scale <= 1e-12))
    if centered_basis:
        # Existing successful records retain their arithmetic. Formerly rejected
        # flat functions use an equivalent constant shift before calibration.
        center, scale = stable_calibration(calibration_samples(centered=True))
    if not np.isfinite(samples).all() or not np.isfinite(scale).all() or np.any(scale <= 0):
        raise ValueError("nonfinite or degenerate calibration; task was not retried")
    x = np.column_stack([rng("input", j).normal(size=task.rows) for j in range(task.features)])
    inputs = torch.tensor(x, dtype=torch.float64, device=device)
    mechanism = {
        "family": recipe.family,
        "version": recipe.version,
        "active": active.tolist(),
        "frequencies": frequencies.tolist(),
        "phases": phases.tolist(),
        "coefficients": coefficients.tolist(),
        "center": center.tolist(),
        "scale": scale.tolist(),
        "normalization": "estimated from independent Gaussian calibration",
        "calibration_rows": recipe.calibration_rows,
        "input": "independent standard normal",
    }
    if centered_basis:
        mechanism["centered_basis"] = True
    scores = evaluate_mechanism(mechanism, inputs)
    return complete_task(
        task, asdict(recipe), mechanism, inputs, scores, seed=seed, task_id=task_id, rng=rng
    )
