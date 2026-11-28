"""Immutable numerical records for source function equations."""

import math
from dataclasses import dataclass

from r0.modules.priors.lib.activation_sampling import finite_number
from r0.modules.priors.lib.weight_geometry import Matrix, matrix_shape


def function_width(value: int) -> None:
    if type(value) is not int or not 1 <= value < 2**63:
        raise ValueError("function width must be a positive integer below 2**63")


@dataclass(frozen=True, slots=True)
class FrozenLinearFunction:
    matrix: Matrix
    version: str = "frozen-linear-function-v1"

    def __post_init__(self) -> None:
        matrix_shape(self.matrix)
        if self.version != "frozen-linear-function-v1":
            raise ValueError("unsupported frozen linear function version")

    @property
    def input_width(self) -> int:
        return len(self.matrix[0])

    @property
    def output_width(self) -> int:
        return len(self.matrix)

    @property
    def parameter_count(self) -> int:
        return self.input_width * self.output_width


@dataclass(frozen=True, slots=True)
class FrozenQuadraticFunction:
    input_width: int
    indices: tuple[int, ...]
    matrices: tuple[Matrix, ...]
    version: str = "frozen-quadratic-function-v1"

    def __post_init__(self) -> None:
        function_width(self.input_width)
        if self.version != "frozen-quadratic-function-v1":
            raise ValueError("unsupported frozen quadratic function version")
        if (
            type(self.indices) is not tuple
            or len(self.indices) != min(self.input_width, 20)
            or any(type(i) is not int or not 0 <= i < self.input_width for i in self.indices)
            or len(set(self.indices)) != len(self.indices)
            or (self.input_width <= 20 and self.indices != tuple(range(self.input_width)))
        ):
            raise ValueError("quadratic indices must preserve the source feature-subset rule")
        k = len(self.indices) + 1
        if (
            type(self.matrices) is not tuple
            or not self.matrices
            or any(matrix_shape(matrix) != (k, k) for matrix in self.matrices)
        ):
            raise ValueError(
                "quadratic matrices require one square form per output, including constant"
            )

    @property
    def output_width(self) -> int:
        return len(self.matrices)

    @property
    def parameter_count(self) -> int:
        return 1 + len(self.indices) + self.output_width * (len(self.indices) + 1) ** 2


@dataclass(frozen=True, slots=True)
class FrozenTreeFunction:
    input_width: int
    features: tuple[tuple[int, ...], ...]
    thresholds: Matrix
    leaves: tuple[Matrix, ...]
    version: str = "frozen-oblivious-tree-function-v1"

    def __post_init__(self) -> None:
        function_width(self.input_width)
        trees, depth = matrix_shape(self.thresholds)
        if (
            self.version != "frozen-oblivious-tree-function-v1"
            or not 1 <= trees <= 128
            or not 1 <= depth <= 7
        ):
            raise ValueError(
                "frozen oblivious trees require their version and source tree/depth bounds"
            )
        if (
            type(self.features) is not tuple
            or len(self.features) != trees
            or any(
                type(row) is not tuple
                or len(row) != depth
                or any(type(i) is not int or not 0 <= i < self.input_width for i in row)
                for row in self.features
            )
        ):
            raise ValueError("tree splits must select one input feature per tree and depth")
        if type(self.leaves) is not tuple or len(self.leaves) != trees:
            raise ValueError("tree leaf tables must match the tree count")
        width = matrix_shape(self.leaves[0])[1]
        if any(matrix_shape(table) != (2**depth, width) for table in self.leaves):
            raise ValueError("tree leaves must cover every split bit pattern and output")

    @property
    def output_width(self) -> int:
        return len(self.leaves[0][0])

    @property
    def parameter_count(self) -> int:
        t, h = len(self.features), len(self.features[0])
        return 1 + 2 * t * h + t * 2**h * self.output_width


@dataclass(frozen=True, slots=True)
class FrozenRFFFunction:
    frequencies: Matrix
    phases: tuple[float, ...]
    coefficients: Matrix
    version: str = "frozen-rff-function-v1"

    def __post_init__(self) -> None:
        _, count = matrix_shape(self.frequencies)
        rows, _ = matrix_shape(self.coefficients)
        if self.version != "frozen-rff-function-v1" or rows != count:
            raise ValueError("Fourier output coefficients must match the frequency count")
        if (
            type(self.phases) is not tuple
            or len(self.phases) != count
            or any(not finite_number(p) or not 0 <= p < 2 * math.pi for p in self.phases)
        ):
            raise ValueError("Fourier phases must lie in [0,2pi) and match all frequencies")

    @property
    def input_width(self) -> int:
        return len(self.frequencies)

    @property
    def output_width(self) -> int:
        return len(self.coefficients[0])

    @property
    def parameter_count(self) -> int:
        return len(self.phases) * (self.input_width + 1 + self.output_width)
