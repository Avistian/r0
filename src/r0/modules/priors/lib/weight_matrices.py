"""Frozen construction records for structural projection matrices."""

from dataclasses import dataclass
from typing import Any

import numpy as np

from r0.modules.priors.lib.activation_sampling import finite_number

type Matrix = tuple[tuple[float, ...], ...]


def matrix_shape(values: Matrix) -> tuple[int, int]:
    if (
        type(values) is not tuple
        or not values
        or any(type(row) is not tuple or not row for row in values)
        or any(len(row) != len(values[0]) for row in values)
        or any(not finite_number(value) for row in values for value in row)
    ):
        raise ValueError("weight factors must be nonempty finite rectangular tuple matrices")
    return len(values), len(values[0])


@dataclass(frozen=True, slots=True)
class FactorizedWeights:
    left: Matrix = ((1.0,),)
    right: Matrix = ((1.0,),)
    diagonal: tuple[float, ...] = (1.0,)
    version: str = "factorized-weights-v1"

    def __post_init__(self) -> None:
        _, rank = matrix_shape(self.left)
        rows, _ = matrix_shape(self.right)
        if self.version != "factorized-weights-v1" or rows != rank:
            raise ValueError("factorized weight dimensions or version disagree")
        if (
            type(self.diagonal) is not tuple
            or len(self.diagonal) != rank
            or any(not finite_number(v) or v < 0 for v in self.diagonal)
        ):
            raise ValueError("factor coefficients must be finite nonnegative values matching rank")

    @property
    def shape(self) -> tuple[int, int]:
        return len(self.left), len(self.right[0])

    @property
    def parameter_count(self) -> int:
        return sum(len(row) for row in (*self.left, *self.right)) + len(self.diagonal)

    def materialize(self) -> Matrix:
        """Evaluate admitted factors; coefficients are not claimed to be singular values."""
        with np.errstate(over="ignore", invalid="ignore"):
            values = (np.asarray(self.left, dtype=np.float64) * self.diagonal) @ np.asarray(
                self.right, dtype=np.float64
            )
        if not np.isfinite(values).all():
            raise ValueError("nonfinite factorized projection; construction was not retried")
        return tuple(tuple(float(value) for value in row) for row in values)


def weight_construction_from_dict(data: dict[str, Any]) -> FactorizedWeights:
    fields = dict(data)
    for name in ("left", "right"):
        if name in fields:
            fields[name] = tuple(tuple(row) for row in fields[name])
    if "diagonal" in fields:
        fields["diagonal"] = tuple(fields["diagonal"])
    return FactorizedWeights(**fields)
