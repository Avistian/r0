"""Tensor output for training; immutable episode conversion for existing consumers."""

from dataclasses import dataclass

import numpy as np
import torch

from r0.modules.episodes import Episode, Feature, Row, Target, TaskType
from r0.modules.priors.lib.training_features import TrainingFeature


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
    feature_schema: tuple[TrainingFeature, ...] = ()
    target_origin: str = "adapter-v1"

    def __post_init__(self) -> None:
        if self.target_origin not in ("adapter-v1", "native-variable-v1"):
            raise ValueError("unsupported training target origin")
        if type(self.feature_schema) is not tuple or (
            self.feature_schema
            and (
                len(self.feature_schema) != self.inputs.shape[1]
                or any(type(f) is not TrainingFeature for f in self.feature_schema)
            )
        ):
            raise ValueError("typed training schema must match input columns")

        if self.target_origin == "native-variable-v1" and not self.feature_schema:
            raise ValueError("native task requires a declared feature schema")
        for index, feature in enumerate(self.feature_schema):
            if feature.kind != "continuous":
                values = self.inputs[:, index]
                if not bool(
                    torch.all(
                        torch.isfinite(values)
                        & (values == values.floor())
                        & (values >= 0)
                        & (values < len(feature.vocabulary))
                    )
                ):
                    raise ValueError("native feature codes disagree with declared vocabulary")
        if (
            self.target_origin == "native-variable-v1"
            and self.classes
            and (
                self.targets.dtype != torch.int64
                or not bool(torch.all((self.targets >= 0) & (self.targets < len(self.classes))))
            )
        ):
            raise ValueError("native targets require bounded int64 category codes")

    def episode(self) -> Episode:
        """Materialize the existing immutable contract only when requested."""
        inputs = np.asarray(self.inputs.detach().cpu().numpy(), dtype=np.float64)
        schema = self.feature_schema or (TrainingFeature(),) * self.inputs.shape[1]
        rows = tuple(
            Row(f"row-{i}", tuple(f.decode(float(v)) for f, v in zip(schema, row, strict=True)))
            for i, row in enumerate(inputs)
        )
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
            tuple(Feature(f"x{i}", feature.kind) for i, feature in enumerate(schema)),
            tuple(rows[i] for i in context),
            tuple(targets[i] for i in context),
            tuple(rows[i] for i in queries),
            tuple(targets[i] for i in queries),
            self.classes,
            (
                "declared native variable vocabulary"
                if self.target_origin == "native-variable-v1"
                else "declared generation recipe"
            )
            if self.classes
            else None,
        )
