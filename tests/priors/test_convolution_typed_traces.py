"""Convolution stages preserve direct typed outputs and captured readouts."""

import json
from dataclasses import asdict, replace

import numpy as np
import pytest
import torch

from r0.modules import priors as p


@pytest.mark.parametrize("kind", ["native", "vector", "converted"])
def test_convolution_root_stages_in_typed_graphs(kind: str):
    function = p.FrozenFeatureConv1D(
        2, 1, (p.FeatureConv1DLayer(p.Conv1DGeometry(1, 1), (((2.0,),),), (0.5,)),)
    )
    equation = p.FunctionalCausalNode((), function, p.JointRootBase("gaussian", 2))
    task = p.LinearRecipe(rows=16, context_rows=8, features=1, signal_features=1, active_features=1)
    if kind == "native":
        recipe = p.NativeCausalRecipe(task, (p.NativeCausalNode(equation),), ((0, 0),), (0, 1))
    elif kind == "vector":
        recipe = p.NativeVectorRecipe(
            task,
            (p.NativeVectorNode(equation, (p.VariableDomain(), p.VariableDomain())),),
            ((0, 0),),
            (0, 1),
        )
    else:
        converter = p.CategoricalConverter(
            "side", 0, 1, ("negative", "positive"), "neighbor_id", ((-1.0,), (1.0,))
        )
        recipe = p.ConvertedCausalRecipe(
            task,
            (p.ConvertedCausalNode(equation, (converter,)),),
            (p.ConvertedReadout(0, capture="side"),),
            p.ConvertedReadout(0, channel=1),
        )
    generated = p.generate_task(recipe, seed=17, task_id=2**54 + 9)
    trace = p.explain_generation(generated, [15, 0, 15])
    nodes = {n["id"]: n for n in trace["nodes"]}
    root = np.array(nodes["root-0"]["values"])
    expected = 2 * root + 0.5
    np.testing.assert_allclose(nodes["convolution-0-layer-0-affine"]["values"], expected)
    np.testing.assert_allclose(generated.targets[[15, 0, 15]].numpy(), expected[:, 1])
    assert nodes["function-0"]["parents"] == ["convolution-0-layer-0-after"]
    assert generated.target_origin == "native-variable-v1"
    restored = p.recipe_from_dict(json.loads(json.dumps(asdict(recipe))))
    assert torch.equal(
        p.generate_task(restored, seed=17, task_id=2**54 + 9).targets, generated.targets
    )


@pytest.mark.parametrize("vector", [False, True])
def test_convolution_class_logits_keep_native_labels_and_encoded_descendants(vector: bool):
    function = p.FrozenFeatureConv1D(
        2,
        1,
        (
            p.FeatureConv1DLayer(
                p.Conv1DGeometry(3, 2),
                (((1.0, 0.0),), ((0.0, 1.0),), ((-1.0, -1.0),)),
                (0.0, 0.0, 0.5),
            ),
        ),
    )
    root_eq = p.FunctionalCausalNode(
        (), p.FrozenLinearFunction(((1.0, 0.0), (0.0, 1.0))), root=p.JointRootBase("gaussian", 2)
    )
    equation = p.FunctionalCausalNode((0,), function)
    domain = p.VariableDomain("categorical", ("a", "b", "c"), "argmax")
    controls = p.LinearRecipe(
        rows=16,
        context_rows=8,
        features=2,
        signal_features=2,
        active_features=2,
        task_type="multiclass",
        classes=3,
    )
    if vector:
        recipe = p.NativeVectorRecipe(
            controls,
            (
                p.NativeVectorNode(root_eq, (p.VariableDomain(), p.VariableDomain())),
                p.NativeVectorNode(equation, (domain,)),
            ),
            ((0, 0), (0, 1)),
            (1, 0),
        )
    else:
        recipe = p.NativeCausalRecipe(
            controls,
            (p.NativeCausalNode(root_eq), p.NativeCausalNode(equation, domain)),
            ((0, 0), (0, 1)),
            (1, 0),
        )
    generated = p.generate_task(recipe, seed=17, task_id=2**54 + 9)
    x = generated.inputs.numpy()
    expected = np.column_stack([x[:, 0], x[:, 1], -x[:, 0] - x[:, 1] + 0.5])
    np.testing.assert_array_equal(generated.targets.numpy(), expected.argmax(1))
    assert generated.targets.dtype == torch.int64
    trace = p.explain_generation(generated, [15, 0, 15])
    nodes = {n["id"]: n for n in trace["nodes"]}
    np.testing.assert_allclose(
        nodes["convolution-1-layer-0-affine"]["values"], expected[[15, 0, 15]]
    )
    assert nodes["function-1"]["parents"] == ["convolution-1-layer-0-after"]

    child = p.FunctionalCausalNode((1,), p.FrozenLinearFunction(((2.0, 5.0, 11.0),)))
    descendant = (
        p.NativeVectorNode(child, (p.VariableDomain(),)) if vector else p.NativeCausalNode(child)
    )
    regression = replace(controls, task_type="regression", classes=0)
    downstream = replace(recipe, task=regression, nodes=(*recipe.nodes, descendant), target=(2, 0))
    actual = p.generate_task(downstream, seed=17, task_id=2**54 + 9)
    np.testing.assert_array_equal(
        actual.targets.numpy(), np.array([2.0, 5.0, 11.0])[expected.argmax(1)]
    )
    assert p.explain_generation(actual, [15, 0, 15])["rows"] == [15, 0, 15]
