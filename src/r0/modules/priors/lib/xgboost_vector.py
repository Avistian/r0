"""Immutable CPU histogram vector-leaf model; distinct from sklearn GBDT."""

import math
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True, slots=True)
class XGBoostVectorTree:
    features: tuple[int, ...]
    thresholds: tuple[float, ...]
    left: tuple[int, ...]
    right: tuple[int, ...]
    default_left: tuple[bool, ...]
    values: tuple[tuple[float, ...], ...]

    def __post_init__(self) -> None:
        arrays = (
            self.features,
            self.thresholds,
            self.left,
            self.right,
            self.default_left,
            self.values,
        )
        if any(type(a) is not tuple for a in arrays):
            raise ValueError("vector tree arrays must be immutable")
        n = len(self.features)
        if not 1 <= n <= 511 or any(len(a) != n for a in arrays):
            raise ValueError("vector tree arrays must align within the node bound")
        incoming = [0] * n
        depths = [0] * n
        width = len(self.values[0])
        if not 1 <= width <= 100:
            raise ValueError("vector leaves require bounded width")
        for i in range(n):
            f, left, right = self.features[i], self.left[i], self.right[i]
            if (
                any(type(v) is not int for v in (f, left, right))
                or type(self.default_left[i]) is not bool
                or type(self.thresholds[i]) is not float
                or not math.isfinite(self.thresholds[i])
                or type(self.values[i]) is not tuple
                or len(self.values[i]) != width
                or any(type(v) is not float or not math.isfinite(v) for v in self.values[i])
            ):
                raise ValueError("invalid vector tree field")
            if f == -1:
                if (left, right) != (-1, -1):
                    raise ValueError("vector leaves cannot have children")
            elif f < 0 or not i < left < n or not i < right < n or left == right:
                raise ValueError("invalid vector split children")
            else:
                for child in (left, right):
                    incoming[child] += 1
                    depths[child] = depths[i] + 1
        if incoming != [0] + [1] * (n - 1) or max(depths) > 8:
            raise ValueError("vector tree must be reachable, acyclic and depth bounded")


@dataclass(frozen=True, slots=True)
class FrozenXGBoostVector:
    input_width: int
    initial: tuple[float, ...]
    trees: tuple[XGBoostVectorTree, ...]
    objective: str = "reg:squarederror"
    backend_version: str = "3.4.1"
    version: str = "frozen-xgboost-vector-v1"

    def __post_init__(self) -> None:
        if (
            type(self.input_width) is not int
            or not 1 <= self.input_width <= 256
            or self.version != "frozen-xgboost-vector-v1"
            or self.backend_version != "3.4.1"
            or self.objective not in ("reg:squarederror", "binary:logistic", "multi:softprob")
            or type(self.initial) is not tuple
            or not 1 <= len(self.initial) <= 100
            or any(type(v) is not float or not math.isfinite(v) for v in self.initial)
            or type(self.trees) is not tuple
            or not 1 <= len(self.trees) <= 32
            or any(type(t) is not XGBoostVectorTree for t in self.trees)
        ):
            raise ValueError("invalid frozen XGBoost vector model")
        if self.objective == "multi:softprob" and len(self.initial) < 2:
            raise ValueError("softmax requires multiple classes")
        for tree in self.trees:
            if len(tree.values[0]) != len(self.initial) or any(
                f >= self.input_width for f in tree.features
            ):
                raise ValueError("vector tree dimensions differ from model")

    @property
    def parameter_count(self) -> int:
        return len(self.initial) + sum(
            len(t.features) * (5 + self.output_width) for t in self.trees
        )

    @property
    def output_width(self) -> int:
        return len(self.initial)


def evaluate_xgboost_vector(
    function: FrozenXGBoostVector,
    inputs: NDArray[np.float64],
    *,
    margin: bool = False,
    max_cells: int = 1_048_576,
) -> NDArray[np.float32]:
    """Backend-compatible float32 strict-less routing, NaN defaults and additions.

    A bounded numerical adapter. Shared graph integration and provenance of fitting
    are separate layers; this entrypoint does not fit or import XGBoost.
    """
    if (
        type(function) is not FrozenXGBoostVector
        or type(margin) is not bool
        or type(max_cells) is not int
        or max_cells < 1
        or type(inputs) is not np.ndarray
        or inputs.ndim != 2
        or len(inputs) < 1
        or inputs.dtype != np.float64
        or inputs.shape[1] != function.input_width
        or inputs.shape[0] * (function.input_width + 4 * function.output_width + 8) > max_cells
    ):
        raise ValueError("XGBoost vector input exceeds geometry or workspace admission")
    with np.errstate(over="ignore"):
        values = inputs.astype(np.float32)
    if np.isinf(values).any():
        raise ValueError("XGBoost vector inputs overflow float32 or contain infinity")
    scores = np.broadcast_to(
        np.array(function.initial, dtype=np.float32), (len(inputs), function.output_width)
    ).copy()
    for tree in function.trees:
        for row in range(len(inputs)):
            node = 0
            while tree.features[node] >= 0:
                value = values[row, tree.features[node]]
                go_left = (
                    tree.default_left[node]
                    if np.isnan(value)
                    else value < np.float32(tree.thresholds[node])
                )
                node = tree.left[node] if go_left else tree.right[node]
            scores[row] += np.array(tree.values[node], dtype=np.float32)
    if not np.isfinite(scores).all():
        raise ValueError("nonfinite vector boosting margin; no retry")
    if not margin:
        if function.objective == "binary:logistic":
            # Stable sigmoid, including extreme finite margins.
            exp = np.exp(-np.abs(scores))
            scores = np.where(scores >= 0, 1 / (1 + exp), exp / (1 + exp))
        elif function.objective == "multi:softprob":
            scores -= scores.max(axis=1, keepdims=True)
            scores = np.exp(scores)
            scores /= scores.sum(axis=1, keepdims=True)
    return scores
