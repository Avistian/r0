"""Linear functions whose adjacent feature groups switch on together."""

from dataclasses import asdict, dataclass, field

import numpy as np
import torch

from r0.modules.priors.lib.inputs import InputDistribution, sample_inputs
from r0.modules.priors.lib.linear import LinearRecipe
from r0.modules.priors.lib.representatives import record_fingerprint
from r0.modules.priors.lib.task_output import complete_task
from r0.modules.priors.lib.training_task import TrainingTask


@dataclass(frozen=True, slots=True)
class GroupedLinearRecipe:
    task: LinearRecipe = field(default_factory=LinearRecipe)
    inputs: InputDistribution = field(default_factory=InputDistribution)
    group_size: int = 2
    calibration_rows: int = 4096
    version: str = "grouped-linear-v1"

    def __post_init__(self) -> None:
        if self.version != "grouped-linear-v1":
            raise ValueError("unsupported grouped linear version")
        if type(self.group_size) is not int or not 1 <= self.group_size <= 256:
            raise ValueError("group_size must be an integer in 1..256")
        if (
            self.task.signal_features % self.group_size
            or self.task.active_features % self.group_size
        ):
            raise ValueError(
                "signal_features and active_features must be whole multiples of group_size"
            )
        if type(self.calibration_rows) is not int or not 256 <= self.calibration_rows <= 16384:
            raise ValueError("calibration rows must be 256..16384")
        if (
            self.inputs.distribution == "student_t"
            and self.inputs.bound is None
            and self.inputs.df <= 2
        ):
            raise ValueError("linear variance requires second input moment: df > 2")


def generate_grouped_linear(
    recipe: GroupedLinearRecipe, *, seed: int, task_id: int, device: str = "cpu"
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

    groups = np.arange(task.signal_features).reshape(-1, recipe.group_size)
    selected = rng("groups").permutation(len(groups))[: task.active_features // recipe.group_size]
    active = groups[selected].ravel()
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
        raw = calibration @ weights
        center, scale = raw.mean(axis=0), raw.std(axis=0)
    if not np.isfinite(raw).all() or not np.isfinite(scale).all() or np.any(scale <= 0):
        raise ValueError("nonfinite or degenerate calibration; task was not retried")
    x = sample_inputs(recipe.inputs, task.rows, task.features, rng, "input")
    inputs = torch.tensor(x, dtype=torch.float64, device=device)
    scores = inputs[:, : task.signal_features] @ torch.tensor(weights, device=device)
    scores = (scores - torch.tensor(center, device=device)) / torch.tensor(scale, device=device)
    mechanism = {
        "family": "grouped_linear",
        "version": recipe.version,
        "groups": groups.tolist(),
        "selected_groups": selected.tolist(),
        "active": active.tolist(),
        "weights": weights.tolist(),
        "center": center.tolist(),
        "scale": scale.tolist(),
        "inputs": asdict(recipe.inputs),
        "calibration_rows": recipe.calibration_rows,
        "normalization": "estimated from an independent generating-distribution sample",
    }
    return complete_task(
        task, asdict(recipe), mechanism, inputs, scores, seed=seed, task_id=task_id, rng=rng
    )
