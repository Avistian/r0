"""Finite immutable matrix geometry shared by construction records."""

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
