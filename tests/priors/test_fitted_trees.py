"""Independent sklearn oracles and persistence checks through the public API."""

import json
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch
from sklearn import ensemble, tree  # pyright: ignore[reportMissingTypeStubs]

from r0.modules.priors import (
    FittedSplitTable,
    FittedTreeOutput,
    FittedTreePlan,
    FrozenFittedTreeFunction,
    FrozenLinearFunction,
    FunctionalCausalNode,
    FunctionalCausalRecipe,
    JointRootBase,
    LinearRecipe,
    evaluate_frozen_function,
    explain_generation,
    fit_fitted_tree,
    frozen_function_from_dict,
    generate_task,
    inspect_fitted_tree_scores,
    recipe_from_dict,
    sample_fitted_graph,
)


@pytest.mark.parametrize("family", ["dt", "extra_trees", "rf", "gbdt"])
@pytest.mark.parametrize("classes", [0, 2, 5])
def test_frozen_predictions_match_independently_refitted_backend(family: str, classes: int) -> None:
    plan = FittedTreePlan(
        family=family,
        task="classification" if classes else "regression",
        input_width=3,
        output_width=2,
        classes=max(2, classes),
    )
    fit = fit_fitted_tree(
        plan,
        input_rng=np.random.default_rng(4),
        target_rng=np.random.default_rng(5),
        backend_rng=np.random.default_rng(6),
    )
    names = {
        "dt": "DecisionTree",
        "extra_trees": "ExtraTrees",
        "rf": "RandomForest",
        "gbdt": "GradientBoosting",
    }
    constructor: Any = getattr(
        tree if family == "dt" else ensemble,
        names[family] + ("Classifier" if classes else "Regressor"),
    )
    probes = np.random.default_rng(77).normal(size=(39, 3))
    probes = np.concatenate((probes, np.asarray(fit.inputs)))
    # Independent backend sees its own refit and probes at exported boundaries,
    # adjacent float64 values, and their rounded float32 neighbors.
    for output in fit.function.outputs:
        for table in output.trees:
            for feature, threshold in zip(table.features, table.thresholds, strict=True):
                if feature >= 0:
                    rows = np.zeros((5, 3))
                    rounded = np.float32(threshold)
                    rows[:, feature] = [
                        threshold,
                        np.nextafter(threshold, -np.inf),
                        np.nextafter(threshold, np.inf),
                        np.nextafter(rounded, np.float32(-np.inf)),
                        np.nextafter(rounded, np.float32(np.inf)),
                    ]
                    probes = np.concatenate((probes, rows))
    expected: list[Any] = []
    scores = inspect_fitted_tree_scores(fit.function, torch.tensor(probes))
    for channel, params in enumerate(fit.backend_parameters):
        model = constructor(**json.loads(params)).fit(
            np.asarray(fit.inputs), np.asarray(fit.targets)[:, channel]
        )
        expected.append(model.predict(probes))
        if classes:
            np.testing.assert_allclose(
                scores[channel].numpy(), model.predict_proba(probes), rtol=1e-13, atol=1e-13
            )
    actual = evaluate_frozen_function(fit.function, torch.tensor(probes))
    np.testing.assert_allclose(actual.numpy(), np.stack(expected, axis=1), rtol=1e-13, atol=1e-13)
    frozen = frozen_function_from_dict(json.loads(json.dumps(asdict(fit.function))))
    assert frozen == fit.function
    assert torch.equal(actual, evaluate_frozen_function(frozen, torch.tensor(probes)))
    assert torch.equal(actual[:37], evaluate_frozen_function(frozen, torch.tensor(probes[:37])))
    permutation = np.random.default_rng(8).permutation(len(probes))
    assert torch.equal(
        actual[permutation], evaluate_frozen_function(frozen, torch.tensor(probes[permutation]))
    )


def test_probability_mean_and_majority_have_distinct_explicit_predictions() -> None:
    def leaf(a: float, b: float) -> FittedSplitTable:
        return FittedSplitTable((-2,), (-2.0,), (-1,), (-1,), ((a, b),))

    trees = (leaf(0.51, 0.49), leaf(0.51, 0.49), leaf(0.01, 0.99))
    probability = FrozenFittedTreeFunction(
        1, (FittedTreeOutput(trees, "probability_mean", (2, 7)),)
    )
    majority = replace(probability, outputs=(replace(probability.outputs[0], combiner="majority"),))
    assert evaluate_frozen_function(probability, torch.zeros((1, 1))).item() == 7
    assert evaluate_frozen_function(majority, torch.zeros((1, 1))).item() == 2


def test_float32_rounding_and_overflow_are_part_of_frozen_contract() -> None:
    table = FittedSplitTable(
        (0, -2, -2), (1.0, -2.0, -2.0), (1, -1, -1), (2, -1, -1), ((0.0,), (3.0,), (5.0,))
    )
    function = FrozenFittedTreeFunction(1, (FittedTreeOutput((table,), "mean"),))
    inputs = torch.tensor([[np.nextafter(1.0, np.inf)], [1.0 + 2**-22]], dtype=torch.float64)
    assert torch.equal(
        evaluate_frozen_function(function, inputs),
        torch.tensor([[3.0], [5.0]], dtype=torch.float64),
    )
    with pytest.raises(ValueError, match="float32"):
        evaluate_frozen_function(function, torch.tensor([[1e100]], dtype=torch.float64))
    with pytest.raises(ValueError, match="children"):
        replace(table, right=(0, -1, -1))


def _skeleton() -> FunctionalCausalRecipe:
    return FunctionalCausalRecipe(
        LinearRecipe(rows=32, features=1, signal_features=1, active_features=1, context_rows=16),
        (
            FunctionalCausalNode(
                (), FrozenLinearFunction(((1.0,),)), root=JointRootBase("gaussian", 1)
            ),
            FunctionalCausalNode((0,), FrozenLinearFunction(((1.0,),))),
            FunctionalCausalNode((1,), FrozenLinearFunction(((1.0,),))),
        ),
        ((0, 0),),
        ((2, 0),),
    )


def test_sampled_ancestor_fit_generation_inspection_and_json_reopen(tmp_path: Path) -> None:
    plans = ((1, FittedTreePlan(family="rf")), (2, FittedTreePlan(family="gbdt")))
    recipe = sample_fitted_graph(
        _skeleton(),
        plans,
        prefix_rng=np.random.default_rng(10),
        input_rng=np.random.default_rng(11),
        target_rng=np.random.default_rng(12),
        backend_rng=np.random.default_rng(13),
    )
    assert recipe.fits[0].prefix_draws
    assert recipe.fits[1].prefix_draws
    assert len(recipe.fits[1].realization.raw_draws) == 2
    reopened = recipe_from_dict(json.loads(json.dumps(asdict(recipe))))
    assert reopened == recipe
    task = generate_task(reopened, seed=3, task_id=8)
    trace = explain_generation(task, [1, 9, 1])
    assert trace["signal_max_error"] == trace["target_max_error"] == 0
    archive = tmp_path / "fitted-task.json"
    archive.write_text(task.recipe_json)
    stored = recipe_from_dict(json.loads(archive.read_text())["recipe"])
    assert stored == recipe
    original = generate_task(stored, seed=3, task_id=8)
    assert torch.equal(original.inputs, task.inputs)
    assert torch.equal(original.scores, task.scores)
    assert len(recipe.fits[0].realization.inputs) != len(task.inputs)


def test_admission_failure_does_not_consume_rng() -> None:
    from r0.modules.priors import CausalLimits

    rngs = [np.random.default_rng(i) for i in range(3)]
    states = [str(r.bit_generator.state) for r in rngs]
    with pytest.raises(ValueError, match="parameter"):
        fit_fitted_tree(
            FittedTreePlan(),
            input_rng=rngs[0],
            target_rng=rngs[1],
            backend_rng=rngs[2],
            limits=CausalLimits(parameters=1),
        )
    assert states == [str(r.bit_generator.state) for r in rngs]


def test_archive_custody_rejects_altered_targets_and_ancestor_rows() -> None:
    recipe = sample_fitted_graph(
        _skeleton(),
        ((1, FittedTreePlan()),),
        prefix_rng=np.random.default_rng(20),
        input_rng=np.random.default_rng(21),
        target_rng=np.random.default_rng(22),
        backend_rng=np.random.default_rng(23),
    )
    for field in ("inputs", "targets"):
        data = json.loads(json.dumps(asdict(recipe)))
        data["fits"][0]["realization"][field][0][0] += 1
        with pytest.raises(ValueError, match="differ"):
            recipe_from_dict(data)


def test_backend_failure_rolls_back_all_construction_streams(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail(*args: Any, **kwargs: Any) -> None:
        raise RuntimeError("synthetic backend failure")

    monkeypatch.setattr(tree.DecisionTreeRegressor, "fit", fail)
    rngs = [np.random.default_rng(i) for i in range(3)]
    states = [str(r.bit_generator.state) for r in rngs]
    with pytest.raises(RuntimeError, match="synthetic backend failure"):
        fit_fitted_tree(
            FittedTreePlan(), input_rng=rngs[0], target_rng=rngs[1], backend_rng=rngs[2]
        )
    assert states == [str(r.bit_generator.state) for r in rngs]


@pytest.mark.parametrize(
    "engine",
    [np.random.PCG64, np.random.PCG64DXSM, np.random.MT19937, np.random.Philox, np.random.SFC64],
)
def test_all_supported_generators_retain_replayable_fit_custody(engine: Any) -> None:
    rngs = [np.random.Generator(engine(i)) for i in range(3)]
    fit = fit_fitted_tree(
        FittedTreePlan(), input_rng=rngs[0], target_rng=rngs[1], backend_rng=rngs[2]
    )
    initial = [snapshot.restore() for snapshot in fit.initial_rng]
    replay = fit_fitted_tree(
        fit.plan, input_rng=initial[0], target_rng=initial[1], backend_rng=initial[2]
    )
    assert replay == fit


def test_single_observed_class_and_ties_use_retained_class_order() -> None:
    table = FittedSplitTable((-2,), (-2.0,), (-1,), (-1,), ((1.0,),))
    function = FrozenFittedTreeFunction(1, (FittedTreeOutput((table,), "probability_mean", (7,)),))
    assert evaluate_frozen_function(function, torch.zeros((1, 1))).item() == 7
    tied = replace(table, values=((0.5, 0.5),))
    function = replace(function, outputs=(FittedTreeOutput((tied,), "probability_mean", (2, 7)),))
    assert evaluate_frozen_function(function, torch.zeros((1, 1))).item() == 2
