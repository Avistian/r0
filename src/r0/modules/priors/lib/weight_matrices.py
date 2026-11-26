"""Frozen construction records for structural projection matrices."""

import math
from dataclasses import dataclass
from decimal import Decimal, localcontext

import numpy as np

from r0.modules.priors.lib.activation_sampling import finite_number
from r0.modules.priors.lib.weight_vectors import (
    PolynomialWeightVector,
    UniformWeightVector,
    WeightVector,
)

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
    exact_terms_limit: int = 1_000_000

    def __post_init__(self) -> None:
        _, rank = matrix_shape(self.left)
        rows, _ = matrix_shape(self.right)
        if self.version != "factorized-weights-v1" or rows != rank:
            raise ValueError("factorized weight dimensions or version disagree")
        if type(self.exact_terms_limit) is not int or not 1 <= self.exact_terms_limit < 2**63:
            raise ValueError("exact factor arithmetic requires a positive term limit below 2**63")
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
        if self._extreme_products():
            return self._exact_product()
        with np.errstate(over="ignore", invalid="ignore"):
            values = (np.asarray(self.left, dtype=np.float64) * self.diagonal) @ np.asarray(
                self.right, dtype=np.float64
            )
        if not np.isfinite(values).all():
            return self._exact_product()
        return tuple(tuple(float(value) for value in row) for row in values)

    def _extreme_products(self) -> bool:
        """Conservative exponent bounds detect unsafe intermediate products or sums."""
        low, high = math.log(np.finfo(np.float64).tiny), math.log(np.finfo(np.float64).max)
        rank = len(self.diagonal)
        for k, coefficient in enumerate(self.diagonal):
            if coefficient == 0:
                continue
            left = [abs(float(row[k])) for row in self.left if row[k] != 0]
            right = [abs(float(v)) for v in self.right[k] if v != 0]
            if not left or not right:
                continue
            shift = math.log(coefficient)
            smallest, largest = math.log(min(left)) + shift, math.log(max(left)) + shift
            if (
                smallest <= low
                or largest >= high
                or smallest + math.log(min(right)) <= low
                or largest + math.log(max(right)) + math.log(rank) >= high
            ):
                return True
        return False

    def _exact_product(self) -> Matrix:
        rows, columns = self.shape
        rank = len(self.diagonal)
        if rows * columns * rank > self.exact_terms_limit:
            raise ValueError("factorized exact arithmetic term limit exceeded")
        # A float64 has at most 309 integer digits and 1074 fractional places.
        # Three exact factors plus the sum's carry digits fit within this bound.
        with localcontext() as context:
            context.prec = 3 * (309 + 1074) + len(str(rank)) + 2
            context.Emin, context.Emax = -999999, 999999
            # Stream terms instead of retaining another full set of wide Decimal factors.
            result = tuple(
                tuple(
                    float(
                        sum(
                            (
                                Decimal.from_float(float(self.left[i][k]))
                                * Decimal.from_float(float(self.diagonal[k]))
                                * Decimal.from_float(float(self.right[k][j]))
                                for k in range(rank)
                            ),
                            Decimal(0),
                        )
                    )
                    for j in range(columns)
                )
                for i in range(rows)
            )
        if any(not math.isfinite(v) for row in result for v in row):
            raise ValueError("nonfinite factorized projection; construction was not retried")
        return result


@dataclass(frozen=True, slots=True)
class ImportanceWeights:
    gaussian: Matrix = ((1.0,),)
    importance: tuple[WeightVector, ...] = (UniformWeightVector(),)
    epsilon: float = 1e-6
    version: str = "importance-weights-v1"

    def __post_init__(self) -> None:
        rows, columns = matrix_shape(self.gaussian)
        if self.version != "importance-weights-v1" or (
            type(self.importance) is not tuple
            or len(self.importance) != rows
            or any(
                type(v) not in (UniformWeightVector, PolynomialWeightVector)
                or len(v.draws) != columns
                for v in self.importance
            )
        ):
            raise ValueError("importance vectors must match every Gaussian matrix row")
        if not finite_number(self.epsilon) or self.epsilon < 0:
            raise ValueError("row normalization epsilon must be finite and nonnegative")

    @property
    def shape(self) -> tuple[int, int]:
        return len(self.gaussian), len(self.gaussian[0])

    @property
    def parameter_count(self) -> int:
        return (
            sum(len(row) for row in self.gaussian)
            + sum(v.parameter_count for v in self.importance)
            + 1
        )

    def materialize(self) -> Matrix:
        return tuple(
            _importance_row(row, vector, self.epsilon)
            for row, vector in zip(self.gaussian, self.importance, strict=True)
        )


def _importance_row(
    gaussian: tuple[float, ...], vector: WeightVector, epsilon: float
) -> tuple[float, ...]:
    weights = vector.values()
    with np.errstate(over="ignore", invalid="ignore", under="ignore"):
        row = np.asarray(gaussian, dtype=np.float64) * weights
    unsafe = not np.isfinite(row).all() or any(
        g != 0 and abs(v) < np.finfo(np.float64).tiny for g, v in zip(gaussian, row, strict=True)
    )
    if not unsafe:
        scale = max(float(np.abs(row).max()), epsilon)
        if scale == 0:
            return (0.0,) * len(row)
        scaled = row / scale
        return tuple(float(v) for v in scaled / (epsilon / scale + np.linalg.norm(scaled)))
    weight_logs = (
        vector.centered_logs(tuple(i for i, g in enumerate(gaussian) if g))
        if epsilon == 0
        else vector.log_values()
    )
    logs = np.asarray(
        [
            math.log(abs(g)) + w if g else -math.inf
            for g, w in zip(gaussian, weight_logs, strict=True)
        ]
    )
    largest = float(logs.max())
    if largest == -math.inf:
        return (0.0,) * len(row)
    log_norm = largest + math.log(float(np.linalg.norm(np.exp(logs - largest))))
    log_denominator = np.logaddexp(log_norm, math.log(epsilon) if epsilon else -math.inf)
    return tuple(float(v) for v in np.sign(gaussian) * np.exp(logs - log_denominator))


@dataclass(frozen=True, slots=True)
class LaplaceKernelWeights:
    left_points: Matrix = ((0.0,),)
    right_points: Matrix = ((0.0,),)
    signs: Matrix = ((1.0,),)
    distance_scale: float = 1.0
    version: str = "laplace-kernel-weights-v1"
    exact_distance_terms_limit: int = 1_000_000

    def __post_init__(self) -> None:
        rows, dimension = matrix_shape(self.left_points)
        columns, right_dimension = matrix_shape(self.right_points)
        if (
            self.version != "laplace-kernel-weights-v1"
            or dimension != right_dimension
            or matrix_shape(self.signs) != (rows, columns)
            or any(v not in (-1, 0, 1) for row in self.signs for v in row)
        ):
            raise ValueError("kernel point dimensions and entry signs must match the projection")
        if not finite_number(self.distance_scale) or self.distance_scale < 0:
            raise ValueError("kernel distance scale must be finite nonnegative")
        if (
            type(self.exact_distance_terms_limit) is not int
            or not 1 <= self.exact_distance_terms_limit < 2**63
        ):
            raise ValueError("exact kernel distance terms require a positive limit below 2**63")

    @property
    def shape(self) -> tuple[int, int]:
        return len(self.left_points), len(self.right_points)

    @property
    def parameter_count(self) -> int:
        return sum(len(row) for row in (*self.left_points, *self.right_points, *self.signs)) + 1

    def materialize(self) -> Matrix:
        return tuple(
            tuple(
                float(sign) * math.exp(-self._scaled_distance(left, right)) if sign else 0.0
                for right, sign in zip(self.right_points, signs, strict=True)
            )
            for left, signs in zip(self.left_points, self.signs, strict=True)
        )

    def _scaled_distance(self, left: tuple[float, ...], right: tuple[float, ...]) -> float:
        if self.distance_scale == 0:
            return 0.0
        distance = math.dist(left, right)
        if math.isfinite(distance):
            return self.distance_scale * distance
        # Opposite large finite coordinates can overflow their difference even
        # when scaling makes the distance small. Scale exact differences first.
        if self.shape[0] * self.shape[1] * len(left) > self.exact_distance_terms_limit:
            raise ValueError("kernel exact distance term limit exceeded")
        with localcontext() as context:
            context.prec = 2 * (309 + 1074) + 2
            context.Emin, context.Emax = -999999, 999999
            scale = Decimal.from_float(float(self.distance_scale))
            differences = (
                float((Decimal.from_float(float(a)) - Decimal.from_float(float(b))) * scale)
                for a, b in zip(left, right, strict=True)
            )
            return math.hypot(*differences)
