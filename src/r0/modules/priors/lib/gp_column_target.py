"""Frozen linear, full quadratic and tanh targets for GP-generated columns."""

import math
from dataclasses import dataclass

import numpy as np

from r0.modules.priors.lib.activation_sampling import finite_number
from r0.modules.priors.lib.causal_limits import CausalLimits
from r0.modules.priors.lib.gp_column_draw import gp_matrix
from r0.modules.priors.lib.joint_function_admission import function_record_cost
from r0.modules.priors.lib.weight_geometry import Matrix, matrix_shape

_DEFAULT_LIMITS = CausalLimits()


@dataclass(frozen=True, slots=True)
class GPColumnTarget:
    family: str
    weights: Matrix
    bias: tuple[float, ...]
    quadratic: tuple[Matrix, ...] = ()
    hidden: Matrix = ()
    hidden_bias: tuple[float, ...] = ()
    version: str = "gp-column-target-r0-v1"

    def __post_init__(self) -> None:
        if any(type(v) is not tuple for v in (self.quadratic, self.hidden, self.hidden_bias)):
            raise ValueError("GP target parameters require frozen tuples")
        width, outputs = matrix_shape(self.weights)
        if self.version != "gp-column-target-r0-v1" or self.family not in (
            "linear",
            "quadratic",
            "neural",
        ):
            raise ValueError("invalid GP column target family/version")
        if (
            type(self.bias) is not tuple
            or len(self.bias) != outputs
            or any(not finite_number(v) for v in self.bias)
        ):
            raise ValueError("GP target bias width differs")
        if self.family == "quadratic":
            if (
                type(self.quadratic) is not tuple
                or len(self.quadratic) != outputs
                or any(matrix_shape(q) != (width, width) for q in self.quadratic)
            ):
                raise ValueError("GP target requires one square quadratic matrix per output")
        elif self.quadratic:
            raise ValueError("unused quadratic parameters")
        if self.family == "neural":
            _, hidden = matrix_shape(self.hidden)
            if (
                hidden != width
                or type(self.hidden_bias) is not tuple
                or len(self.hidden_bias) != hidden
                or any(not finite_number(v) for v in self.hidden_bias)
            ):
                raise ValueError("GP neural target hidden dimensions disagree")
        elif self.hidden or self.hidden_bias:
            raise ValueError("unused neural parameters")

    @property
    def input_width(self) -> int:
        return len(self.hidden) if self.family == "neural" else len(self.weights)

    @property
    def outputs(self) -> int:
        return len(self.bias)


def evaluate_gp_column_target(
    target: GPColumnTarget, values: Matrix, *, limits: CausalLimits = _DEFAULT_LIMITS
) -> Matrix:
    if type(target) is not GPColumnTarget or matrix_shape(values)[1] != target.input_width:
        raise ValueError("GP target input width differs")
    count, storage = function_record_cost(target, limits)
    rows = len(values)
    channels = target.input_width + len(target.weights) + target.outputs
    if (
        rows > 1024
        or channels > limits.total_channels
        or count + 8 * rows * channels > limits.parameters
        or 4 * storage + 128 * rows * channels > limits.working_bytes
    ):
        raise ValueError("GP target evaluation aggregate limit exceeded")
    x = np.asarray(values, dtype=np.float64)
    with np.errstate(over="ignore", invalid="ignore"):
        inputs = (
            np.tanh(x @ np.asarray(target.hidden) + target.hidden_bias)
            if target.family == "neural"
            else x
        )
        result = inputs @ np.asarray(target.weights) + target.bias
        if target.family == "quadratic":
            for channel, q in enumerate(target.quadratic):
                result[:, channel] += np.sum((x @ np.asarray(q)) * x, axis=1)
    if not np.isfinite(result).all():
        raise ValueError("nonfinite GP target; no clipping or retry")
    return gp_matrix(result)


def sample_gp_target(
    family: str, columns: int, outputs: int, hidden: int, rng: np.random.Generator
) -> GPColumnTarget:
    width = hidden if family == "neural" else columns
    weights = gp_matrix(rng.normal(size=(width, outputs)) / math.sqrt(width))
    bias = tuple(float(v) for v in rng.normal(size=outputs))
    quadratic = (
        tuple(gp_matrix(rng.normal(size=(columns, columns)) / columns) for _ in range(outputs))
        if family == "quadratic"
        else ()
    )
    hidden_weights = (
        gp_matrix(rng.normal(size=(columns, hidden)) / math.sqrt(columns))
        if family == "neural"
        else ()
    )
    hidden_bias = tuple(float(v) for v in rng.normal(size=hidden)) if family == "neural" else ()
    return GPColumnTarget(family, weights, bias, quadratic, hidden_weights, hidden_bias)
