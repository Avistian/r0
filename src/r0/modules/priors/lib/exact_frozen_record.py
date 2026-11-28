"""Type-exact comparison of admitted frozen provenance without serialized copies."""

from dataclasses import fields, is_dataclass
from typing import Any, cast


def exact_frozen_equal(expected: Any, actual: Any) -> bool:
    """Python numeric equality alone cannot bind a strict archived construction."""
    if type(expected) is not type(actual):
        return False
    if is_dataclass(expected) and not isinstance(expected, type):
        return all(
            exact_frozen_equal(getattr(expected, field.name), getattr(actual, field.name))
            for field in fields(expected)
        )
    if type(expected) is tuple:
        left = cast(tuple[Any, ...], expected)
        right = cast(tuple[Any, ...], actual)
        return len(left) == len(right) and all(
            exact_frozen_equal(a, b) for a, b in zip(left, right, strict=True)
        )
    return expected == actual
