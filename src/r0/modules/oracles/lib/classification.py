"""Conjugate categorical posterior predictions with explicit prior knowledge."""

import math
from dataclasses import dataclass
from typing import Literal

from r0.modules.episodes import PredictorView


@dataclass(frozen=True, slots=True)
class ClassificationSpec:
    family: Literal["beta_bernoulli", "dirichlet_categorical"]
    classes: tuple[str, ...]
    concentration: tuple[float, ...]
    positive_class: str | None = None

    def __post_init__(self) -> None:
        if self.family not in ("beta_bernoulli", "dirichlet_categorical"):
            raise ValueError("unsupported classification reference family")
        if type(self.classes) is not tuple or any(
            type(c) is not str or not c for c in self.classes
        ):
            raise ValueError("declare immutable nonempty class names")
        if len(set(self.classes)) != len(self.classes) or not 2 <= len(self.classes) <= 16:
            raise ValueError("declare 2..16 unique classes")
        if type(self.concentration) is not tuple or len(self.concentration) != len(self.classes):
            raise ValueError("concentration must align with class vocabulary")
        if any(
            type(a) not in (float, int) or not math.isfinite(a) or a <= 0
            for a in self.concentration
        ):
            raise ValueError("prior concentration must be finite and strictly positive")
        if self.family == "beta_bernoulli":
            if len(self.classes) != 2 or self.positive_class not in self.classes:
                raise ValueError("coin requires two classes and an explicit heads/positive class")
        elif len(self.classes) < 3 or self.positive_class is not None:
            raise ValueError("multiclass requires at least three classes and no positive_class")


@dataclass(frozen=True, slots=True)
class ClassificationPrediction:
    classes: tuple[str, ...]
    posterior: tuple[float, ...]
    probabilities: tuple[tuple[float, ...], ...]
    positive_class: str | None


def _prediction(
    spec: ClassificationSpec, counts: tuple[int, ...], queries: int
) -> ClassificationPrediction:
    if type(queries) is not int or not 1 <= queries <= 64:
        raise ValueError("reference requires 1..64 queries")
    posterior = tuple(a + n for a, n in zip(spec.concentration, counts, strict=True))
    # Scale before summing to avoid overflow for large, finite concentrations.
    scale = max(posterior)
    total = math.fsum(a / scale for a in posterior)
    probabilities = tuple((a / scale) / total for a in posterior)
    if any(not math.isfinite(a) for a in posterior) or any(p <= 0 for p in probabilities):
        raise ValueError("posterior exceeds supported floating-point range")
    return ClassificationPrediction(
        spec.classes, posterior, (probabilities,) * queries, spec.positive_class
    )


def classification_prior(spec: ClassificationSpec, queries: int) -> ClassificationPrediction:
    return _prediction(spec, (0,) * len(spec.classes), queries)


def classification(view: PredictorView, spec: ClassificationSpec) -> ClassificationPrediction:
    expected = "binary" if spec.family == "beta_bernoulli" else "multiclass"
    if view.task_type != expected or view.classes != spec.classes:
        raise ValueError("reference family and class order must match predictor view")
    if len(view.context) > 64:
        raise ValueError("reference accepts at most 64 context rows")
    counts = tuple(view.context_targets.count(c) for c in spec.classes)
    return _prediction(spec, counts, len(view.queries))
