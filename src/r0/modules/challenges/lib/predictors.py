"""Small context solvers and deliberately flawed controls; no scoring-record access."""

import math

from r0.modules.challenges.lib.cases import Family, Shortcut
from r0.modules.episodes import PredictorView, Target, Value


def reference_predictions(view: PredictorView, family: Family) -> tuple[Target, ...]:
    if family == "single-feature-affine":
        return _affine(view)
    if family != "single-feature-lookup":
        raise ValueError("unsupported declared family")
    candidates: list[tuple[Target, ...]] = []
    for column in range(len(view.features)):
        mapping: dict[Value, Target] = {}
        valid = True
        for row, target in zip(view.context, view.context_targets, strict=True):
            value = row.values[column]
            if value in mapping and mapping[value] != target:
                valid = False
            mapping[value] = target
        if valid and all(r.values[column] in mapping for r in view.queries):
            candidates.append(tuple(mapping[r.values[column]] for r in view.queries))
    if len(candidates) != 1:
        raise ValueError("context must identify exactly one supported feature rule")
    return candidates[0]


def shortcut_predictions(view: PredictorView, shortcut: Shortcut) -> tuple[Target, ...]:
    if shortcut == "range-clipping":
        low, high = (
            min(float(y) for y in view.context_targets),
            max(float(y) for y in view.context_targets),
        )
        return tuple(max(low, min(high, float(y))) for y in _affine(view))
    if shortcut == "fixed-label":
        fixed = {"A": "red", "B": "green", "C": "blue"}
        return tuple(fixed[str(row.values[0])] for row in view.queries)
    if shortcut != "fixed-position":
        raise ValueError("unsupported shortcut")
    return tuple("high" if row.values[0] == 1 else "low" for row in view.queries)


def _number(value: Value) -> float:
    if not isinstance(value, (int, float)):
        raise ValueError("affine reference requires observed numeric values")
    return float(value)


def _affine(view: PredictorView) -> tuple[Target, ...]:
    if view.task_type != "regression" or len(view.features) != 1:
        raise ValueError("affine reference supports one regression feature")
    x = [_number(row.values[0]) for row in view.context]
    y = [float(value) for value in view.context_targets]
    different = next((i for i, value in enumerate(x) if value != x[0]), None)
    if different is None:
        raise ValueError("context does not identify an affine slope")
    slope = (y[different] - y[0]) / (x[different] - x[0])
    intercept = y[0] - slope * x[0]
    if not all(
        math.isclose(slope * a + intercept, b, abs_tol=1e-10, rel_tol=1e-10)
        for a, b in zip(x, y, strict=True)
    ):
        raise ValueError("context contradicts declared affine family")
    predictions = tuple(slope * _number(row.values[0]) + intercept for row in view.queries)
    if not all(math.isfinite(value) for value in predictions):
        raise ValueError("affine predictions must be finite")
    return predictions
