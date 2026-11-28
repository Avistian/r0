"""Float32-routing split-table inference and ensemble inspection."""

import torch

from r0.modules.priors.lib.fitted_tree_records import (
    FittedSplitTable,
    FittedTreeOutput,
    FrozenFittedTreeFunction,
)


def _tree(table: FittedSplitTable, inputs: torch.Tensor) -> torch.Tensor:
    device = inputs.device
    indices = torch.zeros(len(inputs), dtype=torch.long, device=device)
    features = torch.tensor(table.features, dtype=torch.long, device=device)
    left = torch.tensor(table.left, dtype=torch.long, device=device)
    right = torch.tensor(table.right, dtype=torch.long, device=device)
    thresholds = torch.tensor(table.thresholds, dtype=torch.float64, device=device)
    rows = torch.arange(len(inputs), device=device)
    for _ in range(8):
        f = features[indices]
        active = f >= 0
        go_left = inputs[rows, f.clamp_min(0)].to(torch.float64) <= thresholds[indices]
        indices = torch.where(active, torch.where(go_left, left[indices], right[indices]), indices)
    return torch.tensor(table.values, dtype=torch.float64, device=device)[indices]


def _finite_scores(scores: torch.Tensor) -> torch.Tensor:
    if not torch.isfinite(scores).all():
        raise ValueError("nonfinite fitted tree scores; no retry")
    return scores


def fitted_output_scores(output: FittedTreeOutput, inputs: torch.Tensor) -> torch.Tensor:
    if output.combiner in ("mean", "probability_mean", "majority"):
        total = torch.zeros(
            (len(inputs), len(output.trees[0].values[0])), dtype=torch.float64, device=inputs.device
        )
        for tree in output.trees:
            value = _tree(tree, inputs)
            if output.combiner == "majority":
                value = torch.nn.functional.one_hot(value.argmax(dim=1), len(output.classes)).to(
                    torch.float64
                )
            total += value / len(output.trees)
        return _finite_scores(total)
    scores = (
        torch.tensor(output.initial, dtype=torch.float64, device=inputs.device)
        .expand(len(inputs), -1)
        .clone()
    )
    for index, tree in enumerate(output.trees):
        scores[:, index % len(output.initial)] += output.learning_rate * _tree(tree, inputs)[:, 0]
    _finite_scores(scores)
    if output.combiner == "boost_sigmoid":
        probability = scores.sigmoid()
        return _finite_scores(torch.cat((1 - probability, probability), dim=1))
    if output.combiner == "boost_softmax":
        return _finite_scores(scores.softmax(dim=1))
    return scores


def evaluate_fitted_tree(function: FrozenFittedTreeFunction, inputs: torch.Tensor) -> torch.Tensor:
    values = inputs.to(torch.float32)
    if not torch.isfinite(values).all():
        raise ValueError("fitted tree float32 input conversion overflow; no retry")
    columns: list[torch.Tensor] = []
    for output in function.outputs:
        scores = fitted_output_scores(output, values)
        if output.classes:
            classes = torch.tensor(output.classes, dtype=torch.float64, device=inputs.device)
            columns.append(classes[scores.argmax(dim=1)])
        else:
            columns.append(scores[:, 0])
    return torch.stack(columns, dim=1)


def inspect_fitted_tree_scores(
    function: FrozenFittedTreeFunction,
    inputs: torch.Tensor,
) -> tuple[torch.Tensor, ...]:
    """Return probabilities/vote fractions or regression predictions per output.

    Public geometry/finite checks use the same admission as normal evaluation.
    Class order and score semantics are the frozen output's classes and combiner.
    """
    from r0.modules.priors.lib.frozen_function_evaluation import evaluate_frozen_function

    if type(function) is not FrozenFittedTreeFunction:
        raise ValueError("fitted score inspection requires its frozen function")
    evaluate_frozen_function(function, inputs)
    return tuple(fitted_output_scores(o, inputs.to(torch.float32)) for o in function.outputs)
