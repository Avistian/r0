"""Addressable Gaussian linear tasks with independently keyed random streams."""

import math
import platform
from dataclasses import asdict, dataclass

import numpy as np
import torch

from r0.modules.episodes import TaskType
from r0.modules.priors.lib.representatives import encode, record_fingerprint
from r0.modules.priors.lib.training_task import TrainingTask


@dataclass(frozen=True, slots=True)
class LinearRecipe:
    rows: int = 64
    features: int = 8
    signal_features: int = 8
    active_features: int = 4
    context_rows: int = 48
    split: int = 0
    noise_ratio: float = 0.0
    task_type: TaskType = "regression"
    classes: int = 0
    separation: float = 1.0
    class_offsets: tuple[float, ...] = ()
    version: str = "linear-v1"

    def __post_init__(self) -> None:
        if self.version != "linear-v1":
            raise ValueError("unsupported linear recipe version")
        integers = (
            self.rows,
            self.features,
            self.signal_features,
            self.active_features,
            self.context_rows,
            self.split,
        )
        if any(type(v) is not int for v in integers) or not (
            16 <= self.rows <= 20_000
            and 1 <= self.active_features <= self.signal_features <= self.features <= 256
            and 1 <= self.context_rows < self.rows
            and 0 <= self.split < 2**63
        ):
            raise ValueError("invalid sizes, active-feature count or split identity")
        if type(self.noise_ratio) not in (float, int) or not (
            math.isfinite(self.noise_ratio) and self.noise_ratio >= 0
        ):
            raise ValueError("noise_ratio must be finite and nonnegative")
        if self.task_type not in ("regression", "binary", "multiclass"):
            raise ValueError("unsupported task type")
        if type(self.classes) is not int or not (
            (self.task_type == "regression" and self.classes == 0)
            or (self.task_type == "binary" and self.classes == 2)
            or (self.task_type == "multiclass" and 3 <= self.classes <= 100)
        ):
            raise ValueError("class count does not match task type")
        if type(self.separation) not in (float, int) or not (
            math.isfinite(self.separation) and self.separation >= 0
        ):
            raise ValueError("separation must be finite and nonnegative")
        outputs = self.classes if self.task_type == "multiclass" else 1
        if (
            type(self.class_offsets) is not tuple
            or (self.class_offsets and len(self.class_offsets) != outputs)
            or any(type(v) not in (float, int) or not math.isfinite(v) for v in self.class_offsets)
        ):
            raise ValueError("class_offsets must be a finite tuple matching the score count")
        if self.task_type == "regression" and (self.class_offsets or self.separation != 1):
            raise ValueError("classification controls do not apply to regression")
        if self.task_type != "regression" and self.noise_ratio != 0:
            raise ValueError("use separation, not observation noise, for classification")


def generate_linear(
    recipe: LinearRecipe, *, seed: int, task_id: int, device: str = "cpu"
) -> TrainingTask:
    """Unit-variance signal; noise_ratio is noise standard deviation / signal std."""
    if any(type(v) is not int or not 0 <= v < 2**63 for v in (seed, task_id)):
        raise ValueError("seed and task_id must be integers in [0, 2**63)")
    if device not in ("cpu", "cuda"):
        raise ValueError("device must be cpu or cuda; no implicit fallback")
    if device == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA is unavailable; select CPU explicitly")

    def rng(*keys: object) -> np.random.Generator:
        return np.random.default_rng(
            int(record_fingerprint((recipe.version, seed, task_id, keys)), 16)
        )

    # Stable per-column streams preserve active inputs when nuisance columns are added.
    x = np.column_stack([rng("input", j).normal(size=recipe.rows) for j in range(recipe.features)])
    order = rng("active").permutation(recipe.signal_features)
    outputs = recipe.classes if recipe.task_type == "multiclass" else 1
    weights = np.column_stack(
        [rng("weights", channel).normal(size=recipe.signal_features) for channel in range(outputs)]
    )
    weights[order[recipe.active_features :], :] = 0
    weights /= np.linalg.norm(weights, axis=0)
    mechanism = {
        "family": "linear",
        "version": recipe.version,
        "weights": weights.tolist(),
        "active": order[: recipe.active_features].tolist(),
        "input": "independent standard normal",
        "expected_signal_variance": 1.0,
    }
    inputs = torch.tensor(x, dtype=torch.float64, device=device)
    coefficients = torch.tensor(weights, dtype=torch.float64, device=device)
    scores = inputs[:, : recipe.signal_features] @ coefficients
    noise = torch.tensor(rng("noise").normal(size=recipe.rows), dtype=torch.float64, device=device)
    targets = scores[:, 0] + recipe.noise_ratio * noise
    probabilities: torch.Tensor | None = None
    classes: tuple[str, ...] = ()
    if recipe.task_type != "regression":
        offsets = torch.tensor(
            recipe.class_offsets or (0.0,) * outputs, dtype=torch.float64, device=device
        )
        logits = recipe.separation * scores + offsets
        if not torch.isfinite(logits).all():
            raise ValueError("nonfinite class scores; task was not retried")
        if recipe.task_type == "binary":
            positive = torch.sigmoid(logits[:, 0])
            probabilities = torch.stack((1 - positive, positive), dim=1)
        else:
            probabilities = torch.softmax(logits, dim=1)
        uniforms = torch.tensor(
            rng("labels").uniform(size=recipe.rows), dtype=torch.float64, device=device
        )
        targets = torch.sum(uniforms[:, None] >= probabilities.cumsum(dim=1)[:, :-1], dim=1)
        classes = tuple(f"class-{i}" for i in range(recipe.classes))
    if not torch.isfinite(targets).all():
        raise ValueError("nonfinite generated targets; task was not retried")
    permutation = torch.tensor(
        rng("split", recipe.split).permutation(recipe.rows), dtype=torch.long, device=device
    )
    config = asdict(recipe)
    source_config = {k: v for k, v in config.items() if k not in ("context_rows", "split")}
    source_id = record_fingerprint((source_config, seed, task_id))
    return TrainingTask(
        source_id,
        record_fingerprint((source_id, recipe.split, recipe.context_rows)),
        recipe.task_type,
        encode(
            {
                "recipe": config,
                "seed": seed,
                "task_id": task_id,
                "execution": {
                    "implementation": "r0.priors.linear-v1",
                    "device": device,
                    "python": platform.python_version(),
                    "numpy": np.__version__,
                    "torch": str(torch.__version__),
                    "cuda": torch.version.cuda,
                    "gpu": torch.cuda.get_device_name() if device == "cuda" else None,
                },
            }
        ),
        encode(mechanism),
        inputs,
        targets,
        scores,
        probabilities,
        permutation[: recipe.context_rows],
        permutation[recipe.context_rows :],
        classes,
    )
