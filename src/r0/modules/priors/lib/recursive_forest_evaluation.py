"""Float64 inclusive-threshold traversal and explicit majority/mean outputs."""

import math
from typing import Any

import numpy as np
from numpy.typing import NDArray

from r0.modules.priors.lib.causal_limits import CausalLimits
from r0.modules.priors.lib.recursive_forest_plan import recursive_effective_limits
from r0.modules.priors.lib.recursive_forest_records import RecursiveForest
from r0.modules.priors.lib.recursive_forest_sampling import SampledRecursiveForest

_DEFAULT_LIMITS = CausalLimits()


def _admit(
    function: RecursiveForest | SampledRecursiveForest,
    inputs: NDArray[Any],
    limits: CausalLimits,
) -> tuple[RecursiveForest, ...]:
    if (
        type(inputs) is not np.ndarray
        or inputs.dtype != np.dtype("float64")
        or inputs.ndim != 2
        or len(inputs) < 1
        or type(limits) is not CausalLimits
    ):
        raise ValueError("recursive inputs must be a nonempty float64 matrix")
    if type(function) is SampledRecursiveForest:
        limits = recursive_effective_limits(function.limits, limits)
        report = function.plan.admit(limits, function.raw_limits)
        if (
            function.plan.lifecycle == "fresh-forest-per-row"
            and len(inputs) != function.plan.populations
        ):
            raise ValueError("fresh forest lifecycle requires its original population row count")
        forests = function.forests
        width, trees = function.plan.input_width, function.plan.estimators
    elif type(function) is RecursiveForest:
        report = function.admit(len(inputs), limits)
        forests = (function,)
        width, trees = function.input_width, len(function.trees)
    else:
        raise ValueError("recursive evaluation requires a frozen forest or sampled population")
    if inputs.shape[1] != width:
        raise ValueError("recursive input width differs")
    # Include traversal path tuples, finite-check mask and retained output simultaneously.
    if (
        report["estimated_working_bytes"] + len(inputs) * (128 * width + 2048 * trees)
        > limits.working_bytes
    ):
        raise ValueError("recursive evaluation aggregate workspace limit exceeded")
    if type(function) is SampledRecursiveForest:
        function.validate_replay(limits=limits)
    if not np.isfinite(inputs).all():
        raise ValueError("recursive inputs must be finite")
    return forests


def _paths(forest: RecursiveForest, row: NDArray[Any]) -> tuple[tuple[int, ...], ...]:
    paths: list[tuple[int, ...]] = []
    for tree in forest.trees:
        path = [0]
        while tree.nodes[path[-1]].left != -1:
            node = tree.nodes[path[-1]]
            path.append(node.left if row[node.feature] <= node.threshold else node.right)
        paths.append(tuple(path))
    return tuple(paths)


def inspect_recursive_forest(
    function: RecursiveForest | SampledRecursiveForest,
    inputs: NDArray[Any],
    *,
    limits: CausalLimits = _DEFAULT_LIMITS,
) -> tuple[tuple[tuple[int, ...], ...], ...]:
    """Original input row order, then tree order, then visited node indices including leaf."""
    forests = _admit(function, inputs, limits)
    return tuple(
        _paths(forests[0] if len(forests) == 1 else forests[i], row) for i, row in enumerate(inputs)
    )


def evaluate_recursive_forest(
    function: RecursiveForest | SampledRecursiveForest,
    inputs: NDArray[Any],
    *,
    limits: CausalLimits = _DEFAULT_LIMITS,
) -> NDArray[Any]:
    """Regression returns float64 (rows,outputs); classification actual int64 (rows,) labels.

    Per-row reference populations bind forests by original row position. They are not
    a single frozen function, nor a claim of row-permutation equivariance.
    """
    forests = _admit(function, inputs, limits)
    first = forests[0]
    result = np.empty(
        (len(inputs), first.output_width) if first.task == "regression" else (len(inputs),),
        dtype=np.float64 if first.task == "regression" else np.int64,
    )
    for i, row in enumerate(inputs):
        forest = forests[0] if len(forests) == 1 else forests[i]
        paths = _paths(forest, row)
        leaves = [tree.nodes[path[-1]] for tree, path in zip(forest.trees, paths, strict=True)]
        if forest.task == "regression":
            for channel in range(forest.output_width):
                result[i, channel] = math.fsum(n.value[channel] / len(leaves) for n in leaves)
        else:
            labels = [n.label for n in leaves]
            counts = [labels.count(c) for c in range(forest.classes)]
            maximum = max(counts)
            tied = [c for c, count in enumerate(counts) if count == maximum]
            if forest.tie_break == "first-tree":
                result[i] = next(label for label in labels if label in tied)
            else:
                result[i] = min(tied) if forest.tie_break == "smallest" else max(tied)
    return result
