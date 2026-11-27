"""Joint vector population roots retain typed stochastic emissions and logical cuts."""

from dataclasses import asdict, replace
from typing import Any

import numpy as np
import pytest
from numpy.typing import NDArray

from r0.modules import priors as p


def vector_plan():
    return p.RecursiveForestPlan(
        input_width=2,
        estimators=1,
        min_depth=1,
        max_depth=1,
        source="r0-extension-v1",
        leaf_mean=(0.0, 0.0),
        leaf_scale=(1.0, 1.0),
        lifecycle="fresh-forest-per-row",
        populations=16,
    )


def controls():
    return p.LinearRecipe(rows=16, context_rows=8, features=2, signal_features=2, active_features=2)


def literal(sample: p.SampledRecursiveForest, inputs: NDArray[Any]):
    result: list[tuple[float, ...]] = []
    for forest, row in zip(sample.forests, inputs, strict=True):
        tree = forest.trees[0]
        node = tree.nodes[0]
        result.append(
            tree.nodes[node.left if row[node.feature] <= node.threshold else node.right].value
        )
    return np.asarray(result)


@pytest.mark.parametrize("family", ["functional", "native", "vector", "converted"])
def test_stochastic_classification_and_root_population_trace(family: str):

    task_controls = replace(controls(), task_type="multiclass", classes=3)
    root = p.FunctionalCausalNode(
        (), p.FrozenLinearFunction(((1.0, 0.0), (0.0, 1.0))), p.JointRootBase("gaussian", 2)
    )
    child = p.FunctionalCausalNode(
        (0,), p.FrozenLinearFunction(((1.0, 0.0), (0.0, 1.0), (-1.0, 1.0)))
    )
    domain = p.VariableDomain("categorical", ("a", "b", "c"), "softmax")
    if family == "functional":
        base = p.FunctionalCausalRecipe(
            task_controls, (root, child), ((0, 0), (0, 1)), ((1, 0), (1, 1), (1, 2))
        )
    elif family == "native":
        base = p.NativeCausalRecipe(
            task_controls,
            (p.NativeCausalNode(root), p.NativeCausalNode(child, domain)),
            ((0, 0), (0, 1)),
            (1, 0),
        )
    elif family == "vector":
        base = p.NativeVectorRecipe(
            task_controls,
            (
                p.NativeVectorNode(root, (p.VariableDomain(),) * 2),
                p.NativeVectorNode(child, (domain,)),
            ),
            ((0, 0), (0, 1)),
            (1, 0),
        )
    else:
        converter = p.CategoricalConverter(
            "label",
            0,
            3,
            domain.vocabulary,
            "softmax_id",
            ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)),
            mean=(0.0,) * 3,
            scale=(1.0,) * 3,
            bias=(0.0,) * 3,
        )
        base = p.ConvertedCausalRecipe(
            task_controls,
            (p.ConvertedCausalNode(root), p.ConvertedCausalNode(child, (converter,))),
            (p.ConvertedReadout(0, channel=0), p.ConvertedReadout(0, channel=1)),
            p.ConvertedReadout(1, capture="label"),
        )
    plan = vector_plan()
    recipe = p.realize_recursive_population_graph(
        p.RecursivePopulationGraphRequest(base, (p.RecursiveNodePopulationPlan(0, plan),)),
        seed=23,
        task_id=7,
    )
    task = p.generate_task(recipe, seed=23, task_id=7)
    np.testing.assert_array_equal(
        task.inputs, literal(recipe.bindings[0].sample, np.asarray(recipe.bindings[0].inputs))
    )
    assert task.probabilities is not None
    logits = np.column_stack(
        [task.inputs[:, 0], task.inputs[:, 1], -task.inputs[:, 0] + task.inputs[:, 1]]
    )
    exp = np.exp(logits - logits.max(axis=1, keepdims=True))
    np.testing.assert_allclose(
        task.probabilities, exp / exp.sum(axis=1, keepdims=True), rtol=1e-14, atol=1e-14
    )
    assert p.recipe_from_dict(asdict(recipe)) == recipe
    graph = p.explain_generation(task, [15, 0, 15])
    nodes = {n["id"]: n for n in graph["nodes"]}
    np.testing.assert_array_equal(
        np.asarray(nodes["target"]["values"])[:, 0], task.targets[[15, 0, 15]]
    )
    assert any("uniform" in n["id"] or n["id"] == "labels" for n in graph["nodes"])
    with pytest.raises(ValueError, match="semantic replay"):
        p.explain_generation(replace(task, probabilities=task.probabilities.float()), [0])


def test_partial_and_full_joint_vector_cuts_preserve_row_owned_joint_mechanism():

    root = p.FunctionalCausalNode(
        (), p.FrozenLinearFunction(((1.0, 0.0), (0.0, 1.0))), p.JointRootBase("gaussian", 2)
    )
    child = p.FunctionalCausalNode((0,), p.FrozenLinearFunction(((2.0, 3.0),)))
    base = p.NativeVectorRecipe(
        controls(),
        (
            p.NativeVectorNode(root, (p.VariableDomain(),) * 2),
            p.NativeVectorNode(child, (p.VariableDomain(),)),
        ),
        ((0, 0), (0, 1)),
        (1, 0),
    )
    request = p.RecursivePopulationGraphRequest(
        base, (p.RecursiveNodePopulationPlan(0, vector_plan()),)
    )
    # Replace the existing two-channel root jointly, not as independent scalar nodes.
    plan = vector_plan()
    recipe = p.realize_recursive_population_graph(
        replace(request, replacements=(p.RecursiveNodePopulationPlan(0, plan),)), seed=23, task_id=7
    )
    baseline = p.generate_task(recipe, seed=23, task_id=7)
    for variables in ((0,), (0, 1)):
        change = p.CausalInterventionRecipe(
            request.base,
            tuple(p.NativeVectorHardIntervention(0, j, value=2.0 + j) for j in variables),
        )
        changed = p.intervene_recursive_population_graph(recipe, change)
        task = p.generate_task(changed, seed=23, task_id=7)
        np.testing.assert_array_equal(task.inputs[:, 0], [2.0] * 16)
        np.testing.assert_array_equal(
            task.inputs[:, 1], [3.0] * 16 if len(variables) == 2 else baseline.inputs[:, 1]
        )
        assert (changed.inputs[0][1] is None) == (len(variables) == 2)
        assert p.recipe_from_dict(asdict(changed)) == changed
        graph = p.explain_generation(task, [15, 0, 15])
        ids = {n["id"] for n in graph["nodes"]}
        assert ("population-0-tree-0" in ids) == (len(variables) == 1)
        assert "dag-0-forced-score:0" in ids
