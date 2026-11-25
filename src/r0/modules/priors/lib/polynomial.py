"""Sparse additive polynomials with explicit population-moment checks."""

from dataclasses import asdict, dataclass, field

import numpy as np
import torch

from r0.modules.priors.lib.calibration import stable_calibration
from r0.modules.priors.lib.inputs import InputDistribution, sample_inputs
from r0.modules.priors.lib.linear import LinearRecipe
from r0.modules.priors.lib.representatives import record_fingerprint
from r0.modules.priors.lib.signals import evaluate_mechanism
from r0.modules.priors.lib.task_output import complete_task
from r0.modules.priors.lib.training_task import TrainingTask


@dataclass(frozen=True, slots=True)
class PolynomialRecipe:
    task: LinearRecipe = field(default_factory=LinearRecipe)
    inputs: InputDistribution = field(default_factory=InputDistribution)
    degree: int = 3
    calibration_rows: int = 4096
    version: str = "polynomial-v1"

    def __post_init__(self) -> None:
        if self.version != "polynomial-v1":
            raise ValueError("unsupported polynomial version")
        if type(self.degree) is not int or not 1 <= self.degree <= 16:
            raise ValueError("polynomial degree must be 1..16")
        if type(self.calibration_rows) is not int or not 256 <= self.calibration_rows <= 16384:
            raise ValueError("calibration rows must be 256..16384")
        if (
            self.inputs.distribution == "student_t"
            and self.inputs.bound is None
            and self.inputs.df <= 2 * self.degree
        ):
            raise ValueError("polynomial variance requires input moment 2*degree: df > 2*degree")


def generate_polynomial(
    recipe: PolynomialRecipe, *, seed: int, task_id: int, device: str = "cpu"
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
    powers = rng("powers").integers(1, recipe.degree + 1, size=task.signal_features)
    # Include the configured degree, rather than merely proposing a maximum.
    powers[active[0]] = recipe.degree
    outputs = task.classes if task.task_type == "multiclass" else 1
    weights = np.column_stack(
        [rng("weights", c).normal(size=task.signal_features) for c in range(outputs)]
    )
    weights[np.setdiff1d(np.arange(task.signal_features), active)] = 0
    weights /= np.linalg.norm(weights, axis=0)
    calibration = sample_inputs(
        recipe.inputs, recipe.calibration_rows, task.signal_features, rng, "calibration"
    )
    with np.errstate(over="ignore", invalid="ignore"):
        raw = (calibration**powers) @ weights
        center, scale = raw.mean(axis=0), raw.std(axis=0)
    if not np.isfinite(raw).all() or not np.isfinite(scale).all() or np.any(scale <= 0):
        # Preserve existing successful records; rescale only a failed estimate.
        center, scale = stable_calibration(raw)
    x = sample_inputs(recipe.inputs, task.rows, task.features, rng, "input")
    inputs = torch.tensor(x, dtype=torch.float64, device=device)
    mechanism = {
        "family": "linear" if recipe.degree == 1 else "additive_polynomial",
        "version": recipe.version,
        "powers": powers.tolist(),
        "weights": weights.tolist(),
        "active": active.tolist(),
        "center": center.tolist(),
        "scale": scale.tolist(),
        "inputs": asdict(recipe.inputs),
        "calibration_rows": recipe.calibration_rows,
        "normalization": "estimated from an independent generating-distribution sample",
    }
    scores = evaluate_mechanism(mechanism, inputs)
    return complete_task(
        task, asdict(recipe), mechanism, inputs, scores, seed=seed, task_id=task_id, rng=rng
    )
