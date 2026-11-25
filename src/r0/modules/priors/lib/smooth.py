"""Bounded sinusoidal and radial functions of features or weighted projections."""

import math
from dataclasses import asdict, dataclass, field

import numpy as np
import torch
from numpy.typing import NDArray

from r0.modules.priors.lib.calibration import stable_calibration
from r0.modules.priors.lib.inputs import InputDistribution, sample_inputs
from r0.modules.priors.lib.linear import LinearRecipe
from r0.modules.priors.lib.representatives import record_fingerprint
from r0.modules.priors.lib.task_output import complete_task
from r0.modules.priors.lib.training_task import TrainingTask


@dataclass(frozen=True, slots=True)
class SmoothRecipe:
    task: LinearRecipe = field(default_factory=LinearRecipe)
    inputs: InputDistribution = field(default_factory=InputDistribution)
    family: str = "radial"
    projection: str = "additive"
    terms: int = 16
    length_scale: float = 1.0
    width_spread: float = 1.0
    calibration_rows: int = 4096
    version: str = "smooth-v1"

    def __post_init__(self) -> None:
        if self.version != "smooth-v1" or self.family not in ("sinusoidal", "radial"):
            raise ValueError("unsupported smooth recipe")
        if self.projection not in ("additive", "ridge"):
            raise ValueError("projection must be additive or ridge")
        if type(self.terms) is not int or not 1 <= self.terms <= 4096:
            raise ValueError("terms must be 1..4096")
        if self.projection == "additive" and self.terms < self.task.active_features:
            raise ValueError("additive functions need at least one term per active feature")
        for value in (self.length_scale, self.width_spread):
            if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
                raise ValueError("width parameters must be finite and positive")
        if self.width_spread < 1:
            raise ValueError("width_spread must be at least 1")
        if type(self.calibration_rows) is not int or not 256 <= self.calibration_rows <= 16384:
            raise ValueError("calibration rows must be 256..16384")


def _raw_scores(
    x: torch.Tensor,
    directions: NDArray[np.float64],
    centers: NDArray[np.float64],
    widths: NDArray[np.float64],
    weights: NDArray[np.float64],
    family: str,
) -> torch.Tensor:
    direction_tensor = torch.tensor(directions, device=x.device)
    center_tensor = torch.tensor(centers, device=x.device)
    width_tensor = torch.tensor(widths, device=x.device)
    weight_tensor = torch.tensor(weights, device=x.device)
    parts: list[torch.Tensor] = []
    for start in range(0, x.shape[0], 256):
        part = x[start : start + 256]
        z = (part @ direction_tensor.T - center_tensor) / width_tensor
        if not torch.isfinite(z).all():
            raise ValueError("nonfinite smooth arguments; task was not retried")
        values = torch.sin(z) if family == "sinusoidal" else torch.exp(-0.5 * z.square())
        parts.append(values @ weight_tensor)
    return torch.cat(parts)


def generate_smooth(
    recipe: SmoothRecipe, *, seed: int, task_id: int, device: str = "cpu"
) -> TrainingTask:
    if any(type(v) is not int or not 0 <= v < 2**63 for v in (seed, task_id)):
        raise ValueError("seed and task_id must be integers in [0, 2**63)")
    if device not in ("cpu", "cuda") or (device == "cuda" and not torch.cuda.is_available()):
        raise ValueError("select an available CPU/CUDA device; no fallback")
    task = recipe.task

    def rng(*keys: object) -> np.random.Generator:
        return np.random.default_rng(
            int(
                record_fingerprint(
                    (recipe.version, recipe.family, recipe.projection, seed, task_id, keys)
                ),
                16,
            )
        )

    active = rng("active").permutation(task.signal_features)[: task.active_features]
    if recipe.projection == "additive":
        directions = np.zeros((recipe.terms, task.signal_features))
        directions[np.arange(recipe.terms), np.resize(active, recipe.terms)] = 1
    else:
        directions = rng("directions").normal(size=(recipe.terms, task.signal_features))
        directions[:, np.setdiff1d(np.arange(task.signal_features), active)] = 0
        directions /= np.linalg.norm(directions, axis=1, keepdims=True)
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        centers = recipe.inputs.location * directions.sum(axis=1) + recipe.inputs.scale * rng(
            "centers"
        ).normal(size=recipe.terms)
        spread = math.log(recipe.width_spread)
        widths = np.exp(
            math.log(recipe.length_scale) + rng("widths").uniform(-spread, spread, recipe.terms)
        )
    if not np.isfinite(centers).all() or not np.isfinite(widths).all() or np.any(widths <= 0):
        raise ValueError("nonfinite or zero sampled widths/centers; task was not retried")
    outputs = task.classes if task.task_type == "multiclass" else 1
    weights = np.column_stack(
        [rng("weights", c).normal(size=recipe.terms) for c in range(outputs)]
    ) / math.sqrt(recipe.terms)
    calibration = sample_inputs(
        recipe.inputs, recipe.calibration_rows, task.signal_features, rng, "calibration"
    )
    raw = _raw_scores(
        torch.tensor(calibration), directions, centers, widths, weights, recipe.family
    )
    center, scale = stable_calibration(np.asarray(raw.numpy(), dtype=np.float64))
    inputs = torch.tensor(
        sample_inputs(recipe.inputs, task.rows, task.features, rng, "input"),
        dtype=torch.float64,
        device=device,
    )
    scores = _raw_scores(
        inputs[:, : task.signal_features], directions, centers, widths, weights, recipe.family
    )
    scores = (scores - torch.tensor(center, device=device)) / torch.tensor(scale, device=device)
    mechanism = {
        "family": f"{recipe.projection}_{recipe.family}",
        "version": recipe.version,
        "directions": directions.tolist(),
        "centers": centers.tolist(),
        "widths": widths.tolist(),
        "weights": weights.tolist(),
        "active": active.tolist(),
        "center": center.tolist(),
        "scale": scale.tolist(),
        "inputs": asdict(recipe.inputs),
        "calibration_rows": recipe.calibration_rows,
        "normalization": "estimated from an independent generating-distribution sample",
    }
    return complete_task(
        task, asdict(recipe), mechanism, inputs, scores, seed=seed, task_id=task_id, rng=rng
    )
