"""Shared inclusive paths, exact leaf means and explicit majority indicators."""

import math
from collections.abc import Callable, Sequence
from typing import Any

import torch

from r0.modules.priors.lib.recursive_forest_function import FrozenRecursiveForestFunction
from r0.modules.priors.lib.recursive_forest_sampling import SampledRecursiveForest


def recursive_function_outputs(
    function: FrozenRecursiveForestFunction,
    inputs: torch.Tensor,
) -> tuple[torch.Tensor, list[list[list[int]]], list[list[int | list[float]]]]:
    """Enclosing graph has admitted all buffers; no row sorting or refitting occurs."""
    if isinstance(function.source, SampledRecursiveForest):
        function.source.validate_replay(limits=function.source.limits)
    forest = function.forest
    rows = inputs.detach().cpu().numpy()
    paths: list[list[list[int]]] = []
    leaves: list[list[int | list[float]]] = []
    values: list[list[float]] = []
    for row in rows:
        row_paths: list[list[int]] = []
        for tree in forest.trees:
            path = [0]
            while tree.nodes[path[-1]].left != -1:
                node = tree.nodes[path[-1]]
                path.append(node.left if row[node.feature] <= node.threshold else node.right)
            row_paths.append(path)
        reached = [t.nodes[path[-1]] for t, path in zip(forest.trees, row_paths, strict=True)]
        paths.append(row_paths)
        if forest.task == "regression":
            leaves.append([list(n.value) for n in reached])
            values.append(
                [
                    math.fsum(n.value[j] / len(reached) for n in reached)
                    for j in range(forest.output_width)
                ]
            )
        else:
            labels = [n.label for n in reached]
            assert all(label is not None for label in labels)
            labels = [int(label) for label in labels if label is not None]
            leaves.append(list(labels))
            counts = [labels.count(c) for c in range(forest.classes)]
            tied = [c for c, count in enumerate(counts) if count == max(counts)]
            winner = (
                next(label for label in labels if label in tied)
                if forest.tie_break == "first-tree"
                else min(tied)
                if forest.tie_break == "smallest"
                else max(tied)
            )
            values.append([float(c == winner) for c in range(forest.classes)])
    return inputs.new_tensor(values), paths, leaves


def recursive_function_steps(
    function: FrozenRecursiveForestFunction,
    inputs: torch.Tensor,
    *,
    prefix: str,
    parents: list[str],
    indices: Sequence[int],
    add: Callable[..., None],
) -> list[str]:
    result, paths, leaves = recursive_function_outputs(function, inputs)
    members: list[str] = []
    for tree in range(len(function.forest.trees)):
        identifier = f"{prefix}-tree-{tree}"
        selected = [leaves[i][tree] for i in indices]
        values = inputs.new_tensor(selected)
        if values.ndim == 1:
            values = values[:, None]
        add(
            identifier,
            f"Recursive tree {tree} reached leaf",
            parents,
            "follow feature <= threshold left, otherwise right; emit reached leaf",
            values,
            {
                "original_rows": list(indices),
                "paths": [paths[i][tree] for i in indices],
                "leaf_semantics": "actual-class-label"
                if function.forest.classes
                else "regression-vector",
            },
        )
        members.append(identifier)
    identifier = f"{prefix}-aggregate"
    parameters: dict[str, Any] = {
        "output_semantics": function.output_semantics,
        "tie_break": function.forest.tie_break,
        "classes": function.forest.classes,
        "lifecycle": "frozen-function",
        "construction": "retained-sampled-raw-receipt"
        if isinstance(function.source, SampledRecursiveForest)
        else "authored",
    }
    add(
        identifier,
        "Recursive forest aggregation",
        members,
        "majority vote -> exact winner indicator (not probabilities)"
        if function.forest.classes
        else "coordinatewise mean of tree leaves",
        result[list(indices)],
        parameters,
    )
    return [identifier]
