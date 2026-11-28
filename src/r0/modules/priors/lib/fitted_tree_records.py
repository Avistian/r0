"""Immutable fitted split tables; inference has no sklearn dependency."""

import math
from dataclasses import dataclass
from typing import Any, cast

from r0.modules.priors.lib.activation_sampling import finite_number


def _floats(values: object) -> bool:
    return (
        type(values) is tuple
        and bool(cast(tuple[Any, ...], values))
        and all(finite_number(v) for v in cast(tuple[Any, ...], values))
    )


@dataclass(frozen=True, slots=True)
class FittedSplitTable:
    features: tuple[int, ...]
    thresholds: tuple[float, ...]
    left: tuple[int, ...]
    right: tuple[int, ...]
    values: tuple[tuple[float, ...], ...]

    def __post_init__(self) -> None:
        if type(self.features) is not tuple:
            raise ValueError("split features must be an immutable tuple")
        n = len(self.features)
        if not 1 <= n <= 511 or any(
            type(v) is not tuple or len(v) != n
            for v in (self.features, self.thresholds, self.left, self.right, self.values)
        ):
            raise ValueError("split tables require aligned immutable arrays of at most 511 nodes")
        if not _floats(self.thresholds) or any(not _floats(v) for v in self.values):
            raise ValueError("split table thresholds and values must be finite")
        width = len(self.values[0])
        incoming = [0] * n
        depths = [0] * n
        for i, (f, left, r) in enumerate(zip(self.features, self.left, self.right, strict=True)):
            if any(type(v) is not int for v in (f, left, r)) or len(self.values[i]) != width:
                raise ValueError("split table indices and value widths are invalid")
            if f == -2:
                if (left, r) != (-1, -1):
                    raise ValueError("leaves require absent children")
            elif f < 0 or not i < left < n or not i < r < n or left == r:
                raise ValueError("split children must follow their parent without cycles")
            else:
                for child in (left, r):
                    incoming[child] += 1
                    depths[child] = depths[i] + 1
        if incoming != [0] + [1] * (n - 1) or max(depths) > 8:
            raise ValueError("split table must be one reachable tree of depth at most eight")


@dataclass(frozen=True, slots=True)
class FittedTreeOutput:
    trees: tuple[FittedSplitTable, ...]
    combiner: str
    classes: tuple[int, ...] = ()
    initial: tuple[float, ...] = (0.0,)
    learning_rate: float = 1.0

    def __post_init__(self) -> None:
        if (
            type(self.trees) is not tuple
            or not 1 <= len(self.trees) <= 3200
            or any(type(t) is not FittedSplitTable for t in self.trees)
            or type(self.classes) is not tuple
            or any(type(c) is not int or c < 0 for c in self.classes)
            or self.classes != tuple(sorted(set(self.classes)))
            or not _floats(self.initial)
            or not finite_number(self.learning_rate)
            or not 0 < self.learning_rate <= 1
        ):
            raise ValueError("invalid fitted ensemble tables, classes or initialization")
        if self.combiner not in (
            "mean",
            "probability_mean",
            "majority",
            "boost_identity",
            "boost_sigmoid",
            "boost_softmax",
        ):
            raise ValueError("unsupported fitted ensemble combiner")
        classifier = self.combiner in (
            "probability_mean",
            "majority",
            "boost_sigmoid",
            "boost_softmax",
        )
        if bool(self.classes) != classifier or len(self.classes) > 100:
            raise ValueError("classifier ensembles require ordered observed classes")
        boosted = self.combiner.startswith("boost_")
        width = len(self.classes) if classifier and not boosted else 1
        if any(len(t.values[0]) != width for t in self.trees):
            raise ValueError("tree leaf widths differ from ensemble semantics")
        scores = len(self.classes) if self.combiner == "boost_softmax" else 1
        if len(self.initial) != scores or len(self.trees) % scores:
            raise ValueError("boosted stage layout and initializer differ")
        if self.combiner == "boost_sigmoid" and len(self.classes) != 2:
            raise ValueError("sigmoid boosting requires two classes")
        if self.combiner == "boost_softmax" and len(self.classes) < 3:
            raise ValueError("softmax boosting requires at least three classes")
        if not boosted and (self.initial != (0.0,) or self.learning_rate != 1.0):
            raise ValueError("averaging ensembles cannot have boosting coefficients")
        if self.combiner in ("probability_mean", "majority"):
            for tree in self.trees:
                for feature, values in zip(tree.features, tree.values, strict=True):
                    if feature == -2 and (min(values) < 0 or not math.isclose(sum(values), 1.0)):
                        raise ValueError("classifier leaves require normalized class probabilities")


@dataclass(frozen=True, slots=True)
class FrozenFittedTreeFunction:
    input_width: int
    outputs: tuple[FittedTreeOutput, ...]
    version: str = "frozen-fitted-tree-function-v1"
    traversal: str = "finite-float32-leq-float64-v1"

    def __post_init__(self) -> None:
        if (
            self.version != "frozen-fitted-tree-function-v1"
            or self.traversal != "finite-float32-leq-float64-v1"
            or type(self.input_width) is not int
            or not 1 <= self.input_width <= 256
            or type(self.outputs) is not tuple
            or not 1 <= len(self.outputs) <= 16
            or any(type(o) is not FittedTreeOutput for o in self.outputs)
        ):
            raise ValueError("invalid fitted function geometry or version")
        trees = tuple(t for o in self.outputs for t in o.trees)
        if len(trees) > 4096 or sum(len(t.features) for t in trees) > 131072:
            raise ValueError("fitted function exceeds retained tree/node bounds")
        if any(f >= self.input_width for t in trees for f in t.features):
            raise ValueError("fitted split feature exceeds input width")

    @property
    def output_width(self) -> int:
        return len(self.outputs)

    @property
    def parameter_count(self) -> int:
        return sum(
            len(o.classes)
            + len(o.initial)
            + 1
            + sum(len(t.features) * (4 + len(t.values[0])) for t in o.trees)
            for o in self.outputs
        )
