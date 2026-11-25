"""Tensor output for training; immutable episode conversion for existing consumers."""

from dataclasses import dataclass

import numpy as np
import torch

from r0.modules.episodes import Episode, Feature, Row, Target, TaskType


@dataclass(frozen=True, slots=True)
class TrainingTask:
    source_task_id: str
    episode_id: str
    task_type: TaskType
    recipe_json: str
    mechanism_json: str
    inputs: torch.Tensor
    targets: torch.Tensor
    scores: torch.Tensor
    probabilities: torch.Tensor | None
    context_indices: torch.Tensor
    query_indices: torch.Tensor
    classes: tuple[str, ...]

    def episode(self) -> Episode:
        """Materialize the existing immutable contract only when requested."""
        inputs = np.asarray(self.inputs.detach().cpu().numpy(), dtype=np.float64)
        rows = tuple(Row(f"row-{i}", tuple(float(v) for v in row)) for i, row in enumerate(inputs))
        values = np.asarray(self.targets.detach().cpu().numpy(), dtype=np.float64)
        targets: tuple[Target, ...] = (
            tuple(self.classes[int(v)] for v in values)
            if self.classes
            else tuple(float(v) for v in values)
        )
        context = [int(i) for i in np.asarray(self.context_indices.cpu().numpy(), dtype=np.int64)]
        queries = [int(i) for i in np.asarray(self.query_indices.cpu().numpy(), dtype=np.int64)]
        return Episode(
            self.source_task_id,
            self.episode_id,
            self.task_type,
            tuple(Feature(f"x{i}", "continuous") for i in range(self.inputs.shape[1])),
            tuple(rows[i] for i in context),
            tuple(targets[i] for i in context),
            tuple(rows[i] for i in queries),
            tuple(targets[i] for i in queries),
            self.classes,
            "declared generation recipe" if self.classes else None,
        )
