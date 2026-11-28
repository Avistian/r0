"""Original-row convolution traces expose affine, activation and residual parents."""

import numpy as np

from r0.modules import priors as p


def test_shared_convolution_trace_retains_literal_stages_and_residual_dependency():
    function = p.FrozenFeatureConv1D(
        3,
        1,
        (
            p.FeatureConv1DLayer(
                p.Conv1DGeometry(1, 1, residual=True),
                (((2.0,),),),
                (1.0,),
                p.CausalActivation("relu"),
            ),
            p.FeatureConv1DLayer(
                p.Conv1DGeometry(1, 2),
                (((1.0, -1.0),),),
                (0.5,),
                p.CausalActivation("tanh"),
                "before",
            ),
        ),
    )
    root = p.FunctionalCausalNode(
        (),
        p.FrozenLinearFunction(((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))),
        root=p.JointRootBase("gaussian", 3),
    )
    recipe = p.FunctionalCausalRecipe(
        p.LinearRecipe(rows=16, context_rows=8, features=3, signal_features=3, active_features=3),
        (root, p.FunctionalCausalNode((0,), function)),
        ((0, 0), (0, 1), (0, 2)),
        ((1, 0),),
    )
    task = p.generate_task(recipe, seed=23, task_id=2**54 + 13)
    trace = p.explain_generation(task, [15, 0, 15])
    nodes = {n["id"]: n for n in trace["nodes"]}
    x = task.inputs.numpy()[[15, 0, 15]]
    affine = 2 * x + 1
    activated = np.maximum(affine, 0)
    residual = activated + x
    prefix = "convolution-1-layer-"
    np.testing.assert_allclose(nodes[prefix + "0-affine"]["values"], affine)
    np.testing.assert_allclose(nodes[prefix + "0-after"]["values"], activated)
    np.testing.assert_allclose(nodes[prefix + "0-residual"]["values"], residual)
    assert nodes[prefix + "0-residual"]["parents"] == [prefix + "0-after", "causal-0"]
    np.testing.assert_allclose(nodes[prefix + "1-before"]["values"], np.tanh(residual))
    final = np.tanh(residual[:, :-1]) - np.tanh(residual[:, 1:]) + 0.5
    np.testing.assert_allclose(nodes[prefix + "1-affine"]["values"], final)
    assert nodes["function-1"]["parents"] == [prefix + "1-affine"]
    assert nodes[prefix + "1-affine"]["parameters"]["original_rows"] == [15, 0, 15]


def test_parent_convolution_traces_keep_separate_slices_and_joint_input():
    function = p.FrozenFeatureConv1D(
        2, 1, (p.FeatureConv1DLayer(p.Conv1DGeometry(1, 2), (((2.0, -1.0),),), (0.5,)),)
    )
    joint = p.FrozenFeatureConv1D(
        4, 1, (p.FeatureConv1DLayer(p.Conv1DGeometry(1, 4), (((1.0, 2.0, 3.0, 4.0),),), (0.0,)),)
    )
    root = p.FunctionalCausalNode(
        (), p.FrozenLinearFunction(((1.0, 0.0), (0.0, 1.0))), root=p.JointRootBase("gaussian", 2)
    )
    composed = p.FrozenParentFunction((function, function), "sum", joint=joint)
    recipe = p.FunctionalCausalRecipe(
        p.LinearRecipe(rows=16, context_rows=8, features=4, signal_features=4, active_features=4),
        (root, root, p.FunctionalCausalNode((0, 1), composed)),
        ((0, 0), (0, 1), (1, 0), (1, 1)),
        ((2, 0),),
    )
    task = p.generate_task(recipe, seed=19, task_id=2**54 + 7)
    trace = p.explain_generation(task, [15, 0, 15])
    nodes = {n["id"]: n for n in trace["nodes"]}
    x = task.inputs.numpy()[[15, 0, 15]]
    for index in range(2):
        key = f"parent-member-2-{index}"
        np.testing.assert_allclose(
            nodes[key + "-layer-0-affine"]["values"],
            (2 * x[:, 2 * index] - x[:, 2 * index + 1] + 0.5)[:, None],
        )
        assert nodes[key]["parents"] == [key + "-layer-0-after"]
    np.testing.assert_allclose(
        nodes["parent-joint-2-layer-0-affine"]["values"],
        (x @ np.array([1.0, 2.0, 3.0, 4.0]))[:, None],
    )
    assert nodes["parent-joint-2"]["parents"] == ["parent-joint-2-layer-0-after"]


def test_soft_replacement_trace_uses_effective_convolution_and_existing_parents():
    original = p.FrozenFeatureConv1D(
        2, 1, (p.FeatureConv1DLayer(p.Conv1DGeometry(1, 2), (((1.0, 1.0),),), (0.0,)),)
    )
    changed = p.FrozenFeatureConv1D(
        2, 1, (p.FeatureConv1DLayer(p.Conv1DGeometry(1, 2), (((3.0, -2.0),),), (1.0,)),)
    )
    root = p.FunctionalCausalNode(
        (), p.FrozenLinearFunction(((1.0, 0.0), (0.0, 1.0))), root=p.JointRootBase("gaussian", 2)
    )
    recipe = p.FunctionalCausalRecipe(
        p.LinearRecipe(rows=16, context_rows=8, features=2, signal_features=2, active_features=2),
        (root, p.FunctionalCausalNode((0,), original)),
        ((0, 0), (0, 1)),
        ((1, 0),),
    )
    pair = p.generate_causal_pair(
        recipe, interventions=(p.SoftIntervention(1, function=changed),), seed=31, task_id=2**54 + 9
    )
    trace = p.explain_generation(pair.after, [15, 0, 15])
    nodes = {n["id"]: n for n in trace["nodes"]}
    x = pair.after.inputs.numpy()[[15, 0, 15]]
    np.testing.assert_allclose(
        nodes["convolution-1-layer-0-affine"]["values"], (3 * x[:, 0] - 2 * x[:, 1] + 1)[:, None]
    )
    assert nodes["intervention-1-function"]["parents"] == ["convolution-1-layer-0-after"]
    seen: set[str] = set()
    for node in trace["nodes"]:
        if node["id"].startswith("convolution-"):
            assert set(node["parents"]) <= seen
        seen.add(node["id"])
