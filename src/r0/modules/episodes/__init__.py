"""Validated immutable prediction episodes and their inference-only view."""

import math
from dataclasses import dataclass, replace
from typing import Literal

FeatureKind = Literal["continuous", "count", "binary", "categorical", "ordinal", "timestamp"]
TaskType = Literal["regression", "binary", "multiclass"]
Value = float | int | str | None
Target = float | int | str


@dataclass(frozen=True, slots=True)
class Feature:
    name: str
    kind: FeatureKind
    order: tuple[str, ...] = ()
    unit: str | None = None
    timezone: str | None = None

    def __post_init__(self) -> None:
        if (
            type(self.name) is not str
            or not self.name
            or self.kind
            not in (
                "continuous",
                "count",
                "binary",
                "categorical",
                "ordinal",
                "timestamp",
            )
        ):
            raise ValueError("feature must have a name and a supported kind")
        if type(self.order) is not tuple or any(type(v) is not str for v in self.order):
            raise ValueError("category order must be a tuple of strings")
        if self.kind == "ordinal":
            if not self.order or len(set(self.order)) != len(self.order):
                raise ValueError("ordinal feature requires a unique declared order")
        elif self.order:
            raise ValueError("only ordinal features accept declared order")
        if self.kind == "timestamp":
            if self.unit not in ("s", "ms", "us", "ns") or self.timezone != "UTC":
                raise ValueError("timestamps require epoch unit s/ms/us/ns and timezone UTC")
        elif self.unit is not None or self.timezone is not None:
            raise ValueError("unit and timezone apply only to timestamps")


@dataclass(frozen=True, slots=True)
class Row:
    id: str
    values: tuple[Value, ...]

    def __post_init__(self) -> None:
        if type(self.id) is not str or not self.id:
            raise ValueError("row identity must be a nonempty string")
        if type(self.values) is not tuple:
            raise ValueError("row values must be an immutable tuple")
        for value in self.values:
            if value is not None and type(value) not in (str, int, float):
                raise ValueError("unsupported feature value")
            if isinstance(value, float) and not math.isfinite(value):
                raise ValueError("numeric values must be finite; use None for missing values")


def _validate_rows(features: tuple[Feature, ...], rows: tuple[Row, ...]) -> None:
    for row in rows:
        if len(row.values) != len(features):
            raise ValueError(f"row {row.id}: inconsistent feature schema")
        for feature, value in zip(features, row.values, strict=True):
            if value is None:
                continue
            if feature.kind in ("categorical", "ordinal"):
                if not isinstance(value, str):
                    raise ValueError(f"{feature.name}: category values must be strings")
                if feature.kind == "ordinal" and value not in feature.order:
                    raise ValueError(f"{feature.name}: value outside declared category order")
            elif not isinstance(value, (int, float)):
                raise ValueError(f"{feature.name}: numeric feature requires a number")
            elif feature.kind == "count" and (value < 0 or value != int(value)):
                raise ValueError(f"{feature.name}: count must be a nonnegative integer")
            elif feature.kind == "binary" and value not in (0, 1):
                raise ValueError(f"{feature.name}: binary value must be zero or one")


@dataclass(frozen=True, slots=True)
class PredictorView:
    """Only inference-available fields; all nested values are immutable."""

    task_type: TaskType
    features: tuple[Feature, ...]
    context: tuple[Row, ...]
    context_targets: tuple[Target, ...]
    queries: tuple[Row, ...]
    classes: tuple[str, ...]
    class_provenance: str | None

    def __post_init__(self) -> None:
        _validate_view(self)


@dataclass(frozen=True, slots=True)
class Episode:
    source_task_id: str
    episode_id: str
    task_type: TaskType
    features: tuple[Feature, ...]
    context: tuple[Row, ...]
    context_targets: tuple[Target, ...]
    queries: tuple[Row, ...]
    query_targets: tuple[Target, ...]
    classes: tuple[str, ...] = ()
    class_provenance: str | None = None

    def __post_init__(self) -> None:
        if any(type(v) is not str or not v for v in (self.source_task_id, self.episode_id)):
            raise ValueError("source task and episode identities must be nonempty strings")
        _validate_view(self)

    def predictor_view(self) -> PredictorView:
        return PredictorView(
            self.task_type,
            tuple(replace(feature) for feature in self.features),
            tuple(replace(row) for row in self.context),
            self.context_targets,
            tuple(replace(row) for row in self.queries),
            self.classes,
            self.class_provenance,
        )


def _validate_view(self: Episode | PredictorView) -> None:
    if self.task_type not in ("regression", "binary", "multiclass"):
        raise ValueError("unsupported task type")
    for value in (
        self.features,
        self.context,
        self.queries,
        self.context_targets,
        self.classes,
    ):
        if type(value) is not tuple:
            raise ValueError("episode collections must be immutable tuples")
    if not self.features or not self.context or not self.queries:
        raise ValueError("episode requires features, context rows and query rows")
    if len({f.name for f in self.features}) != len(self.features):
        raise ValueError("feature names must be unique")
    rows = self.context + self.queries
    if len({r.id for r in rows}) != len(rows):
        raise ValueError("context/query row identities must be unique and disjoint")
    _validate_rows(self.features, rows)
    if len(self.context) != len(self.context_targets):
        raise ValueError("every context row must have a target")
    targets = self.context_targets
    if isinstance(self, Episode):
        if type(self.query_targets) is not tuple or len(self.queries) != len(self.query_targets):
            raise ValueError("every scored row must have a target")
        targets += self.query_targets
    if self.task_type == "regression":
        if self.classes or self.class_provenance is not None:
            raise ValueError("regression must not declare a class vocabulary")
        if any(type(y) not in (float, int) or not math.isfinite(float(y)) for y in targets):
            raise ValueError("regression targets must be finite numbers")
    else:
        if any(type(c) is not str or not c for c in self.classes):
            raise ValueError("class vocabulary must contain nonempty strings")
        minimum = 2 if self.task_type == "binary" else 3
        if len(self.classes) < minimum or len(set(self.classes)) != len(self.classes):
            raise ValueError("declare a unique class vocabulary of sufficient size")
        if self.task_type == "binary" and len(self.classes) != 2:
            raise ValueError("binary tasks require exactly two classes")
        if type(self.class_provenance) is not str or not self.class_provenance:
            raise ValueError("class vocabulary provenance is required")
        if any(y not in self.classes for y in targets):
            raise ValueError("target outside declared class vocabulary")
