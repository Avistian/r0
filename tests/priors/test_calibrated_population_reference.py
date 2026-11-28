"""Original-row references apply retained calibration before the target adapter."""

from dataclasses import asdict
from types import FrameType
from typing import Any, Literal

import numpy as np
import pytest

from r0.modules import priors as p


def test_calibrated_reference_retains_source_and_fixed_fit():
    cap = p.CausalLimits(parameters=16_000_000, working_bytes=4 * 1024**3)
    task = p.LinearRecipe(rows=16, context_rows=8, features=1, signal_features=1, active_features=1)
    f = p.FrozenLinearFunction(((1.0,),))
    base = p.FunctionalCausalRecipe(
        task,
        (
            p.FunctionalCausalNode((), f, p.JointRootBase("gaussian", 1)),
            p.FunctionalCausalNode((0,), f),
        ),
        ((0, 0),),
        ((1, 0),),
        limits=cap,
    )
    forest = p.RecursiveForestPlan(
        input_width=1,
        intervals=((-1.0, 1.0),),
        estimators=1,
        min_depth=1,
        max_depth=1,
        lifecycle="fresh-forest-per-row",
        populations=16,
    )
    original = p.realize_recursive_population_graph(
        p.RecursivePopulationGraphRequest(
            base, (p.RecursiveNodePopulationPlan(1, forest),), limits=cap
        ),
        seed=31,
        task_id=41,
    )
    fitted = p.calibrate_population_graph(
        original, calibration_rows=32, calibration_seed=71, calibration_task_id=73
    )
    reference = p.evaluate_population_graph_reference(
        fitted, row=3, feature=0, values=(-1.0, 0.0, 1.0), limits=cap
    )
    assert reference.version == "calibrated-population-graph-reference-v1"
    fit = p.generate_task(fitted.calibration_population, seed=71, task_id=73).scores.numpy()
    raw = p.evaluate_population_graph_reference(
        original, row=3, feature=0, values=(-1.0, 0.0, 1.0), limits=cap
    )
    np.testing.assert_allclose(
        np.array(reference.scores), (np.array(raw.scores) - fit.mean(0)) / fit.std(0), atol=1e-13
    )
    assert p.population_graph_reference_from_dict(asdict(reference), limits=cap) == reference


def _source(
    family: str,
    task_type: Literal["regression", "binary", "multiclass"] = "regression",
    noise_ratio: float = 0.0,
):
    width = 3 if task_type == "multiclass" else 1
    cap = p.CausalLimits(parameters=32_000_000, working_bytes=8 * 1024**3)
    task = p.LinearRecipe(
        rows=16,
        context_rows=8,
        features=1,
        signal_features=1,
        active_features=1,
        task_type=task_type,
        noise_ratio=noise_ratio,
        classes=0 if task_type == "regression" else (2 if task_type == "binary" else 3),
    )
    nodes = (
        p.FunctionalCausalNode(
            (), p.FrozenLinearFunction(((1.0,),)), p.JointRootBase("gaussian", 1)
        ),
        p.FunctionalCausalNode((0,), p.FrozenLinearFunction(tuple((1.0,) for _ in range(width)))),
        p.FunctionalCausalNode(
            (1,),
            p.FrozenLinearFunction(
                tuple(tuple(3.0 if i == j else 0.0 for j in range(width)) for i in range(width))
            ),
        ),
    )
    if family == "functional":
        base = p.FunctionalCausalRecipe(
            task, nodes, ((0, 0),), tuple((2, j) for j in range(width)), limits=cap
        )
    elif family == "native":
        base = p.NativeCausalRecipe(
            task, tuple(p.NativeCausalNode(n) for n in nodes), ((0, 0),), (2, 0), limits=cap
        )
    elif family == "vector":
        base = p.NativeVectorRecipe(
            task,
            tuple(p.NativeVectorNode(n, (p.VariableDomain(),)) for n in nodes),
            ((0, 0),),
            (2, 0),
            limits=cap,
        )
    else:
        base = p.ConvertedCausalRecipe(
            task,
            tuple(p.ConvertedCausalNode(n) for n in nodes),
            (p.ConvertedReadout(0, channel=0),),
            p.ConvertedReadout(2, channel=0),
            limits=cap,
        )
    forest = p.RecursiveForestPlan(
        input_width=1,
        intervals=((-1.0, 1.0),),
        estimators=2,
        min_depth=1,
        max_depth=2,
        lifecycle="fresh-forest-per-row",
        populations=16,
        source="r0-extension-v1",
        leaf_mean=(0.1,) * width,
        leaf_scale=(1.0,) * width,
    )
    original = p.realize_recursive_population_graph(
        p.RecursivePopulationGraphRequest(
            base, (p.RecursiveNodePopulationPlan(1, forest),), limits=cap
        ),
        seed=31,
        task_id=2**55 + 41,
    )
    return p.calibrate_population_graph(
        original, calibration_rows=32, calibration_seed=71, calibration_task_id=73
    )


def _literal(forest: p.RecursiveForest, value: float):
    leaves: list[tuple[float, ...]] = []
    for tree in forest.trees:
        index = 0
        while tree.nodes[index].left != -1:
            node = tree.nodes[index]
            index = node.left if value <= node.threshold else node.right
        leaves.append(tree.nodes[index].value)
    return np.mean(leaves, axis=0)


@pytest.mark.parametrize("family", ["functional", "native", "vector", "converted"])
@pytest.mark.parametrize("environment", ["original", "query", "matched", "population"])
def test_fixed_reference_literal_all_families(
    family: str, environment: Literal["original", "query", "matched", "population"]
):
    fitted = _source(family)
    source: (
        p.CalibratedPopulationGraph
        | p.CalibratedPopulationQuery
        | p.CalibratedPopulationIntervention
    ) = fitted
    if environment == "query":
        source = p.realize_calibrated_population_query(fitted, seed=101, task_id=103)
    factor = 3.0
    if environment in ("matched", "population"):
        change = p.CausalInterventionRecipe(
            fitted.original.request.base,
            (p.SoftIntervention(2, function=p.FrozenLinearFunction(((4.7,),))),),
            mode=environment,
            population=11,
        )
        source = p.intervene_calibrated_population(fitted, change)
        factor = 4.7
    values = (-1.0, 0.0, 1.0)
    result = p.evaluate_population_graph_reference(
        source, row=3, feature=0, values=values, limits=fitted.limits
    )
    forests = (
        source.original.samples[0].forests
        if isinstance(source, p.CalibratedPopulationIntervention)
        else source.original.bindings[0].sample.forests
    )
    raw = np.array([factor * _literal(forests[3], v) for v in values])
    binding = fitted.calibration_population.bindings[0]
    reference = np.array(
        [3 * _literal(f, x[0]) for f, x in zip(binding.sample.forests, binding.inputs, strict=True)]
    )
    expected = (raw - reference.mean(0)) / reference.std(0)
    assert isinstance(result, p.CalibratedPopulationReference)
    np.testing.assert_allclose(result.raw_scores, raw, atol=1e-13)
    np.testing.assert_allclose(result.scores, expected, atol=1e-13)
    np.testing.assert_allclose(result.targets, expected[:, 0], atol=1e-13)
    assert p.population_graph_reference_from_dict(asdict(result), limits=fitted.limits) == result


@pytest.mark.parametrize("task_type", ["binary", "multiclass"])
@pytest.mark.parametrize("environment", ["matched", "population"])
def test_calibrated_reference_probabilities_follow_fixed_scores(
    task_type: Literal["binary", "multiclass"], environment: Literal["matched", "population"]
):
    fitted = _source("functional", task_type)
    change = p.CausalInterventionRecipe(
        fitted.original.request.base,
        (p.SoftIntervention(0, function=p.FrozenLinearFunction(((1.0,),))),),
        mode=environment,
        population=13,
    )
    source = p.intervene_calibrated_population(fitted, change)
    task = p.generate_task(source, seed=31, task_id=2**55 + 41)
    anchor = float(task.inputs[3, 0])
    result = p.evaluate_population_graph_reference(
        source, row=3, feature=0, values=(-1.0, anchor, 1.0), limits=fitted.limits
    )
    scores = np.array(result.scores)
    if task_type == "binary":
        positive = 1 / (1 + np.exp(-scores[:, 0]))
        expected = np.column_stack((1 - positive, positive))
    else:
        exp = np.exp(scores - scores.max(1, keepdims=True))
        expected = exp / exp.sum(1, keepdims=True)
    assert result.probabilities is not None
    np.testing.assert_allclose(result.probabilities, expected, atol=1e-13)
    assert result.targets[1] == int(task.targets[3])
    assert task.probabilities is not None
    assert result.probabilities is not None
    np.testing.assert_array_equal(result.probabilities[1], task.probabilities[3].numpy())


@pytest.mark.parametrize("mutation", ["raw_scores", "fingerprint", "identity", "grid", "budget"])
def test_reference_metadata_rejects_before_source_restore(mutation: str):
    from typing import Any
    from unittest.mock import patch

    source = _source("functional")
    result = p.evaluate_population_graph_reference(
        source, row=3, feature=0, values=(-1.0, 0.0, 1.0), limits=source.limits
    )
    raw: dict[str, Any] = asdict(result)
    cap = source.limits
    if mutation == "raw_scores":
        raw["raw_scores"] = ((True,),) * 3
    elif mutation == "fingerprint":
        raw["calibration_fingerprint"] = "0" * 64
    elif mutation == "identity":
        raw["task_id"] = 41
    elif mutation == "grid":
        raw["values"] = (float("nan"), 0.0, 1.0)
    else:
        cap = p.CausalLimits(parameters=1)
    cls = type(source.original.bindings[0].sample.initial)
    with (
        patch.object(
            cls, "restore", side_effect=AssertionError("malformed reference restored RNG")
        ),
        pytest.raises(ValueError, match="reference|identity|limit|finite|metadata"),
    ):
        p.population_graph_reference_from_dict(raw, limits=cap)


def test_reference_never_constructs_a_new_fit_population():
    import sys

    source = _source("functional")
    calls: list[str] = []
    codes = (p.calibrate_population_graph.__code__, p.realize_recursive_population_graph.__code__)
    previous = sys.getprofile()

    def profile(frame: FrameType, event: str, arg: Any) -> None:
        if event == "call" and frame.f_code in codes:
            calls.append(frame.f_code.co_name)

    try:
        sys.setprofile(profile)
        result = p.evaluate_population_graph_reference(
            source, row=3, feature=0, values=(-1.0, 0.0, 1.0), limits=source.limits
        )
    finally:
        sys.setprofile(previous)
    assert isinstance(result, p.CalibratedPopulationReference)
    assert result.calibration_fingerprint
    assert calls == []


@pytest.mark.parametrize("environment", ["original", "population"])
def test_reference_regression_noise_keeps_original_row_draw(environment: str):
    fitted = _source("functional", noise_ratio=0.6)
    source: p.CalibratedPopulationGraph | p.CalibratedPopulationIntervention = fitted
    if environment == "population":
        source = p.intervene_calibrated_population(
            fitted,
            p.CausalInterventionRecipe(
                fitted.original.request.base,
                (p.SoftIntervention(0, function=p.FrozenLinearFunction(((1.0,),))),),
                mode="population",
                population=19,
            ),
        )
    original = p.generate_task(source, seed=31, task_id=2**55 + 41)
    anchor = float(original.inputs[3, 0])
    reference = p.evaluate_population_graph_reference(
        source, row=3, feature=0, values=(-1.0, anchor, 1.0), limits=fitted.limits
    )
    residual = float(original.targets[3] - original.scores[3, 0])
    np.testing.assert_allclose(
        np.array(reference.targets) - np.array(reference.scores)[:, 0], residual, atol=1e-13
    )
    assert reference.targets[1] == float(original.targets[3])
