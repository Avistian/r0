"""Compare numeric report leaves to an independently authored reference card."""

import math
from fractions import Fraction
from typing import cast


def _child(value: object, part: str) -> object:
    if isinstance(value, dict):
        return cast(dict[str, object], value)[part]
    if isinstance(value, (tuple, list)) and part.isdecimal():
        return cast(tuple[object, ...], value)[int(part)]
    raise KeyError(part)


def _represent(value: Fraction) -> float | None:
    try:
        return float(value)
    except OverflowError:
        return None


def compare_oracle_report(
    report: dict[str, object], expected: dict[str, float], *, atol: float, rtol: float
) -> dict[str, object]:
    if not expected:
        raise ValueError("reference card must contain at least one expected value")
    if any(type(t) not in (int, float) or not math.isfinite(t) or t < 0 for t in (atol, rtol)):
        raise ValueError("reference tolerances must be finite and nonnegative")
    comparisons: list[dict[str, object]] = []
    for path, target in sorted(expected.items()):
        if type(path) is not str or type(target) not in (int, float) or not math.isfinite(target):
            raise ValueError("reference card must map dotted paths to finite numeric answers")
        actual: object = report
        try:
            for part in path.split("."):
                actual = _child(actual, part)
        except (KeyError, IndexError) as error:
            raise ValueError(f"invalid reference path: {path}") from error
        if type(actual) not in (int, float):
            raise ValueError(f"reference path is not numeric: {path}")
        observed = float(cast(float, actual))
        if not math.isfinite(observed):
            raise ValueError(f"reference path is not finite: {path}")
        # Exact rational arithmetic over the supplied floats avoids losing the verdict
        # when subtraction, multiplication or relative-error reporting would overflow.
        difference = abs(Fraction(observed) - Fraction(target))
        threshold = Fraction(atol) + Fraction(rtol) * abs(Fraction(target))
        absolute_error = _represent(difference)
        allowed_error = _represent(threshold)
        relative_error = _represent(difference / abs(Fraction(target))) if target else None
        comparisons.append(
            {
                "path": path,
                "expected": target,
                "observed": observed,
                "absolute_error": absolute_error,
                "absolute_error_reason": None
                if absolute_error is not None
                else "exceeds float64 range",
                "relative_error": relative_error,
                "relative_error_reason": (
                    None
                    if relative_error is not None
                    else "zero expected value"
                    if target == 0
                    else "exceeds float64 range"
                ),
                "allowed_error": allowed_error,
                "allowed_error_reason": None
                if allowed_error is not None
                else "exceeds float64 range",
                "passed": difference <= threshold,
            }
        )
    return {
        "status": "PASS" if all(c["passed"] for c in comparisons) else "FAIL",
        "atol": atol,
        "rtol": rtol,
        "comparisons": comparisons,
    }
