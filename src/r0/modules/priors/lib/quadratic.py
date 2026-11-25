"""Quadratic factors with independently controlled off-diagonal interactions."""

import math
from dataclasses import asdict, dataclass, field

import numpy as np
import torch

from r0.modules.priors.lib.inputs import InputDistribution, sample_inputs
from r0.modules.priors.lib.linear import LinearRecipe
from r0.modules.priors.lib.representatives import record_fingerprint
from r0.modules.priors.lib.task_output import complete_task
from r0.modules.priors.lib.training_task import TrainingTask


@dataclass(frozen=True, slots=True)
class QuadraticRecipe:
    task: LinearRecipe = field(default_factory=LinearRecipe)
    inputs: InputDistribution = field(default_factory=InputDistribution)
    rank: int = 8
    interaction_strength: float = 1.0
    linear_strength: float = 1.0
    calibration_rows: int = 4096
    version: str = "quadratic-v1"

    def __post_init__(self) -> None:
        if self.version != "quadratic-v1":
            raise ValueError("unsupported quadratic version")
        if type(self.rank) is not int or not 1 <= self.rank <= 256:
            raise ValueError("quadratic rank must be 1..256")
        for value in (self.interaction_strength, self.linear_strength):
            if type(value) not in (int, float) or not math.isfinite(value):
                raise ValueError("quadratic strengths must be finite numbers")
        if self.linear_strength < 0:
            raise ValueError("linear strength must be nonnegative")
        if type(self.calibration_rows) is not int or not 256 <= self.calibration_rows <= 16384:
            raise ValueError("calibration rows must be 256..16384")
        if (
            self.inputs.distribution == "student_t"
            and self.inputs.bound is None
            and self.inputs.df <= 4
        ):
            raise ValueError("quadratic variance requires fourth input moment: df > 4")


def generate_quadratic(
    recipe: QuadraticRecipe, *, seed: int, task_id: int, device: str = "cpu"
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
    inactive = np.setdiff1d(np.arange(task.signal_features), active)
    outputs = task.classes if task.task_type == "multiclass" else 1
    directions = rng("directions").normal(size=(recipe.rank, task.signal_features))
    directions[:, inactive] = 0
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)
    weights = np.column_stack([rng("weights", c).normal(size=recipe.rank) for c in range(outputs)])
    weights /= math.sqrt(recipe.rank)
    linear = np.column_stack(
        [rng("linear", c).normal(size=task.signal_features) for c in range(outputs)]
    )
    linear[inactive] = 0
    linear /= np.linalg.norm(linear, axis=0)
    diagonal = (directions**2).T @ weights
    calibration = sample_inputs(
        recipe.inputs, recipe.calibration_rows, task.signal_features, rng, "calibration"
    )
    strength = recipe.interaction_strength
    with np.errstate(over="ignore", invalid="ignore"):
        raw = (
            strength * ((calibration @ directions.T) ** 2 @ weights)
            + (1 - strength) * (calibration**2 @ diagonal)
            + recipe.linear_strength * (calibration @ linear)
        )
        center, scale = raw.mean(axis=0), raw.std(axis=0)
    if not np.isfinite(raw).all() or not np.isfinite(scale).all() or np.any(scale <= 1e-12):
        raise ValueError("nonfinite or degenerate calibration; task was not retried")
    x = sample_inputs(recipe.inputs, task.rows, task.features, rng, "input")
    inputs = torch.tensor(x, dtype=torch.float64, device=device)
    signal = inputs[:, : task.signal_features]
    direction_tensor = torch.tensor(directions, device=device)
    weight_tensor = torch.tensor(weights, device=device)
    scores = (
        strength * ((signal @ direction_tensor.T) ** 2 @ weight_tensor)
        + (1 - strength) * (signal**2 @ torch.tensor(diagonal, device=device))
        + recipe.linear_strength * (signal @ torch.tensor(linear, device=device))
    )
    scores = (scores - torch.tensor(center, device=device)) / torch.tensor(scale, device=device)
    mechanism = {
        "family": "quadratic",
        "version": recipe.version,
        "directions": directions.tolist(),
        "weights": weights.tolist(),
        "linear_weights": linear.tolist(),
        "active": active.tolist(),
        "interaction_strength": strength,
        "linear_strength": recipe.linear_strength,
        "center": center.tolist(),
        "scale": scale.tolist(),
        "inputs": asdict(recipe.inputs),
        "calibration_rows": recipe.calibration_rows,
        "normalization": "estimated from an independent generating-distribution sample",
    }
    return complete_task(
        task, asdict(recipe), mechanism, inputs, scores, seed=seed, task_id=task_id, rng=rng
    )
