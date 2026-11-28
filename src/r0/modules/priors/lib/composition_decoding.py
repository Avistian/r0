"""Decode retained composite expression trees without changing their operations."""

from typing import Any

from r0.modules.priors.lib.compositions import Combination


def expression_from_dict(value: Any) -> int | Combination:
    if type(value) is int:
        return value
    fields = dict(value)
    fields["children"] = tuple(expression_from_dict(child) for child in fields["children"])
    return Combination(**fields)
