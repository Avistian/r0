"""Context-fitted fixture preprocessing; missingness is never imputed implicitly."""

import math
from dataclasses import dataclass
from statistics import mean as arithmetic_mean
from statistics import pstdev

from r0.modules.episodes import Feature, PredictorView, Row, Target, TaskType


@dataclass(frozen=True)
class ColumnStatistics:
    name: str
    mean: float
    scale: float
    categories: tuple[str, ...]
    all_missing: bool
    constant: bool


@dataclass(frozen=True)
class EncodedRow:
    id: str
    values: tuple[float | int, ...]
    missing: tuple[bool, ...]
    unknown: tuple[bool, ...]


@dataclass(frozen=True)
class TransformedView:
    recipe: str
    task_type: TaskType
    features: tuple[Feature, ...]
    context: tuple[EncodedRow, ...]
    context_targets: tuple[Target, ...]
    queries: tuple[EncodedRow, ...]
    classes: tuple[str, ...]
    class_provenance: str | None
    statistics: tuple[ColumnStatistics, ...]


def transform(view: PredictorView) -> TransformedView:
    """Fit only context, then transform context and queries with the same recipe."""
    statistics: list[ColumnStatistics] = []
    for column, feature in enumerate(view.features):
        observed = [row.values[column] for row in view.context if row.values[column] is not None]
        mean, scale = 0.0, 1.0
        categories: tuple[str, ...] = ()
        if feature.kind in ("continuous", "count") and observed:
            numbers = [float(v) for v in observed if isinstance(v, (int, float))]
            mean = arithmetic_mean(numbers)
            scale = pstdev(numbers) or 1.0
            if not math.isfinite(mean) or not math.isfinite(scale):
                raise ValueError(f"{feature.name}: context statistics exceed numeric range")
        if feature.kind == "categorical":
            categories = tuple(sorted({v for v in observed if isinstance(v, str)}))
        elif feature.kind == "ordinal":
            categories = feature.order
        statistics.append(
            ColumnStatistics(
                feature.name,
                mean,
                scale,
                categories,
                not observed,
                bool(observed) and len(set(observed)) == 1,
            )
        )

    def encode(rows: tuple[Row, ...]) -> tuple[EncodedRow, ...]:
        encoded: list[EncodedRow] = []
        for row in rows:
            values: list[float | int] = []
            missing: list[bool] = []
            unknown: list[bool] = []
            for feature, stat, value in zip(view.features, statistics, row.values, strict=True):
                missing.append(value is None)
                unseen = isinstance(value, str) and value not in stat.categories
                unknown.append(unseen)
                if value is None:
                    output: float | int = 0
                elif isinstance(value, str):
                    output = -1 if unseen else stat.categories.index(value)
                elif feature.kind in ("continuous", "count"):
                    output = (value - stat.mean) / stat.scale
                else:
                    output = value
                if not math.isfinite(output):
                    raise ValueError(f"{feature.name}: transformed value exceeds numeric range")
                values.append(output)
            encoded.append(EncodedRow(row.id, tuple(values), tuple(missing), tuple(unknown)))
        return tuple(encoded)

    return TransformedView(
        "context-v1",
        view.task_type,
        view.features,
        encode(view.context),
        view.context_targets,
        encode(view.queries),
        view.classes,
        view.class_provenance,
        tuple(statistics),
    )
