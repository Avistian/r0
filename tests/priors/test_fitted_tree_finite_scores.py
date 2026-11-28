"""Immutable fitted tables cannot hide nonfinite arithmetic behind class argmax."""

import pytest
import torch

from r0.modules import priors


def function(combiner: str, *, overflow: bool = False) -> priors.FrozenFittedTreeFunction:
    classes = (0, 1, 2) if combiner == "boost_softmax" else (0, 1)
    if combiner in ("mean", "boost_identity"):
        classes = ()
    values = (
        (0.25, 0.75)
        if combiner in ("probability_mean", "majority")
        else (1e308 if overflow else 1.0,)
    )
    leaf = priors.FittedSplitTable((-2,), (-2.0,), (-1,), (-1,), (values,))
    width = 3 if combiner == "boost_softmax" else 1
    output = priors.FittedTreeOutput(
        (leaf,) * (2 * width), combiner, classes=classes, initial=(0.0,) * width
    )
    return priors.FrozenFittedTreeFunction(1, (output,))


@pytest.mark.parametrize("combiner", ["boost_identity", "boost_sigmoid", "boost_softmax"])
@pytest.mark.parametrize("inspect", [False, True])
def test_nonfinite_boosted_scores_reject_before_prediction(combiner: str, inspect: bool) -> None:
    evaluate = priors.inspect_fitted_tree_scores if inspect else priors.evaluate_frozen_function
    with pytest.raises(ValueError, match="nonfinite"):
        evaluate(function(combiner, overflow=True), torch.zeros((1, 1), dtype=torch.float64))


@pytest.mark.parametrize(
    "combiner",
    ["mean", "probability_mean", "majority", "boost_identity", "boost_sigmoid", "boost_softmax"],
)
def test_finite_combiner_semantics_and_float32_overflow(combiner: str) -> None:
    fitted = function(combiner)
    inputs = torch.tensor([[0.0], [1.0]], dtype=torch.float64)
    actual = priors.evaluate_frozen_function(fitted, inputs)
    scores = priors.inspect_fitted_tree_scores(fitted, inputs)[0]
    if combiner == "mean":
        expected = [[1.0], [1.0]]
    elif combiner == "boost_identity":
        expected = [[2.0], [2.0]]
    else:
        expected = [[0.0], [0.0]] if combiner == "boost_softmax" else [[1.0], [1.0]]
        torch.testing.assert_close(scores.sum(dim=1), torch.ones(2, dtype=torch.float64))
    torch.testing.assert_close(actual, torch.tensor(expected, dtype=torch.float64))
    assert bool(torch.isfinite(scores).all())
    for evaluate in (priors.evaluate_frozen_function, priors.inspect_fitted_tree_scores):
        with pytest.raises(ValueError, match="float32 input conversion overflow"):
            evaluate(fitted, torch.tensor([[1e100]], dtype=torch.float64))
