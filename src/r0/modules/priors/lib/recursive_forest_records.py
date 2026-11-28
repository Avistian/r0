"""Immutable path-conditioned recursive trees, distinct from oblivious/fitted trees."""

import math
from dataclasses import dataclass

from r0.modules.priors.lib.causal_limits import CausalLimits
from r0.modules.priors.lib.joint_function_admission import function_record_cost


@dataclass(frozen=True, slots=True)
class RecursiveTreeNode:
    feature: int
    threshold: float
    interval: tuple[float, float]
    depth: int
    left: int = -1
    right: int = -1
    value: tuple[float, ...] = ()
    label: int | None = None

    def __post_init__(self) -> None:
        if (
            type(self.feature) is not int
            or not 0 <= self.feature < 256
            or type(self.threshold) is not float
            or not math.isfinite(self.threshold)
            or type(self.interval) is not tuple
            or len(self.interval) != 2
            or any(type(v) is not float or not math.isfinite(v) for v in self.interval)
            or not self.interval[0] <= self.threshold <= self.interval[1]
            or type(self.depth) is not int
            or not 0 <= self.depth <= 10
            or type(self.left) is not int
            or type(self.right) is not int
            or type(self.value) is not tuple
            or len(self.value) > 32
            or any(type(v) is not float or not math.isfinite(v) for v in self.value)
            or self.label is not None
            and (type(self.label) is not int or not 0 <= self.label < 100)
        ):
            raise ValueError("invalid frozen recursive node")
        if self.left == -1 and self.right == -1:
            if bool(self.value) == (self.label is not None):
                raise ValueError("recursive leaf needs exactly one value or label")
        elif self.left < 0 or self.right < 0 or self.value or self.label is not None:
            raise ValueError("recursive split requires two children and no output")


@dataclass(frozen=True, slots=True)
class RecursiveTree:
    nodes: tuple[RecursiveTreeNode, ...]

    def __post_init__(self) -> None:
        if (
            type(self.nodes) is not tuple
            or not 1 <= len(self.nodes) <= 2047
            or any(type(n) is not RecursiveTreeNode for n in self.nodes)
        ):
            raise ValueError("recursive tree requires bounded immutable nodes")
        incoming = [0] * len(self.nodes)
        if self.nodes[0].depth != 0:
            raise ValueError("recursive root depth must be zero")
        for index, node in enumerate(self.nodes):
            if node.left == -1:
                continue
            if not index < node.left < len(self.nodes) or not index < node.right < len(self.nodes):
                raise ValueError("recursive children must follow parent")
            for child in (node.left, node.right):
                incoming[child] += 1
                if self.nodes[child].depth != node.depth + 1:
                    raise ValueError("recursive child depth differs")
        if incoming != [0] + [1] * (len(self.nodes) - 1):
            raise ValueError("recursive tree must be fully reachable without shared children")


@dataclass(frozen=True, slots=True)
class RecursiveForest:
    input_width: int
    trees: tuple[RecursiveTree, ...]
    task: str = "regression"
    output_width: int = 1
    classes: int = 0
    tie_break: str = "none"
    version: str = "frozen-recursive-forest-v1"

    def __post_init__(self) -> None:
        if (
            self.version != "frozen-recursive-forest-v1"
            or type(self.input_width) is not int
            or not 1 <= self.input_width <= 256
            or type(self.output_width) is not int
            or not 1 <= self.output_width <= 32
            or type(self.classes) is not int
            or type(self.trees) is not tuple
            or not 1 <= len(self.trees) <= 64
            or any(type(t) is not RecursiveTree for t in self.trees)
        ):
            raise ValueError("invalid frozen recursive forest")
        if self.task == "regression":
            if self.classes != 0 or self.tie_break != "none":
                raise ValueError("regression forest cannot retain class controls")
        elif self.task == "classification":
            if (
                self.output_width != 1
                or not 2 <= self.classes <= 100
                or self.tie_break not in ("smallest", "largest", "first-tree")
            ):
                raise ValueError("classification forest requires explicit majority tie policy")
        else:
            raise ValueError("invalid recursive forest task")
        for tree in self.trees:
            paths: list[tuple[int, dict[int, tuple[float, float]]]] = [(0, {})]
            while paths:
                index, intervals = paths.pop()
                node = tree.nodes[index]
                if node.feature >= self.input_width or (
                    node.feature in intervals and intervals[node.feature] != node.interval
                ):
                    raise ValueError("recursive node violates ancestor interval or feature width")
                if node.left == -1:
                    if (
                        self.task == "regression"
                        and len(node.value) != self.output_width
                        or self.task == "classification"
                        and (node.label is None or node.label >= self.classes)
                    ):
                        raise ValueError("recursive leaf output differs from forest task")
                else:
                    for child, interval in (
                        (node.left, (node.interval[0], node.threshold)),
                        (node.right, (node.threshold, node.interval[1])),
                    ):
                        paths.append((child, {**intervals, node.feature: interval}))

    def admit(self, rows: int, limits: CausalLimits) -> dict[str, int]:
        if type(rows) is not int or not 1 <= rows < 2**63 or type(limits) is not CausalLimits:
            raise ValueError("recursive evaluation requires positive rows and frozen limits")
        n = sum(len(t.nodes) for t in self.trees)
        numbers, storage = function_record_cost(self, limits)
        working = (
            4 * storage
            + 128 * n * self.input_width
            + 8
            * rows
            * (self.input_width + len(self.trees) * (self.output_width + 2) + self.classes + 8)
        )
        if (
            n > limits.nodes
            or max(self.input_width, self.output_width) > limits.node_channels
            or self.input_width + self.output_width > limits.total_channels
            or working > limits.working_bytes
        ):
            raise ValueError("recursive forest evaluation geometry or workspace limit exceeded")
        return {"nodes": n, "parameters": numbers, "estimated_working_bytes": working}
