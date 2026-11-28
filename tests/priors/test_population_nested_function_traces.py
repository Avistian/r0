"""Detailed frozen function stages coexist with original-row population mechanisms."""

import numpy as np
import pytest

from r0.modules import priors as p


def test_conv_population_linear_trace_has_literal_original_row_stages():
    conv = p.FrozenFeatureConv1D(
        2, 1, (p.FeatureConv1DLayer(p.Conv1DGeometry(1, 2), (((2.0, -1.0),),), (0.5,)),)
    )
    base = p.FunctionalCausalRecipe(
        p.LinearRecipe(rows=16, context_rows=7, features=1, signal_features=1, active_features=1),
        (
            p.FunctionalCausalNode((), conv, p.JointRootBase("gaussian", 2)),
            p.FunctionalCausalNode((0,), p.FrozenLinearFunction(((1.0,),))),
            p.FunctionalCausalNode((1,), p.FrozenLinearFunction(((3.0,),))),
        ),
        ((0, 0),),
        ((2, 0),),
    )
    plan = p.RecursiveForestPlan(
        input_width=1,
        intervals=((-1.0, 1.0),),
        estimators=1,
        min_depth=1,
        max_depth=1,
        lifecycle="fresh-forest-per-row",
        populations=16,
    )
    recipe = p.realize_recursive_population_graph(
        p.RecursivePopulationGraphRequest(base, (p.RecursiveNodePopulationPlan(1, plan),)),
        seed=41,
        task_id=2**55 + 7,
    )
    task = p.generate_task(recipe, seed=41, task_id=2**55 + 7)
    rows = [15, 2, 15]
    graph = p.explain_generation(task, rows)
    nodes = {n["id"]: n for n in graph["nodes"]}
    affine = nodes["dag-0-detail-layer-0-affine"]
    inputs = np.asarray(nodes["dag-0-inputs"]["values"])
    expected = 2 * inputs[:, 0] - inputs[:, 1] + 0.5
    np.testing.assert_allclose(np.asarray(affine["values"])[:, 0], expected, rtol=0, atol=1e-15)
    assert affine["parameters"]["original_rows"] == rows
    assert affine["parameters"]["scope"] == "original dataset rows"
    leaves: list[float] = []
    for row in rows:
        tree = recipe.bindings[0].sample.forests[row].trees[0]
        split = tree.nodes[0]
        value = recipe.bindings[0].inputs[row][0]
        leaf = tree.nodes[split.left if value <= split.threshold else split.right]
        leaves.append(3 * leaf.value[0])
    np.testing.assert_array_equal(task.targets[rows], leaves)
    assert nodes["dag-0-function"]["parents"] == ["dag-0-detail-layer-0-after"]
    seen: set[str] = set()
    for node in graph["nodes"]:
        assert set(node["parents"]) <= seen
        assert node["id"] not in seen
        seen.add(node["id"])


@pytest.mark.parametrize("family", ["functional", "converted", "native", "vector"])
def test_nested_parent_conv_network_product_stages_and_shared_scope(family: str):
    layer = p.Conv1DLayerPrior((p.Conv1DGeometry(1, 1),), bias_law="rademacher")
    conv = p.sample_feature_conv1d(
        p.FeatureConv1DRequest(2, 1, (layer,)),
        named_rng=np.random.default_rng(3),
        raw_rng=np.random.default_rng(5),
    )
    network = p.FrozenNetworkFunction(
        2, (p.NetworkBlock((0,), (p.NetworkAffine(((2.0, 0.0), (0.0, -1.0)), (1.0, 0.5)),)),)
    )
    product = p.FrozenKumoProductFunction(
        p.FrozenLinearFunction(((1.0, 0.0, 0.0, 0.0), (0.0, 1.0, 0.0, 0.0))),
        p.FrozenLinearFunction(((0.0, 0.0, 1.0, 0.0), (0.0, 0.0, 0.0, 1.0))),
    )
    parent = p.FrozenParentFunction((conv, network), aggregation="mean", joint=product)
    root = p.FunctionalCausalNode(
        (), p.FrozenLinearFunction(((1.0, 0.0), (0.0, 1.0))), p.JointRootBase("gaussian", 2)
    )
    base = p.FunctionalCausalRecipe(
        p.LinearRecipe(rows=16, context_rows=7, features=2, signal_features=2, active_features=2),
        (
            root,
            root,
            p.FunctionalCausalNode((0, 1), parent),
            p.FunctionalCausalNode((2,), p.FrozenLinearFunction(((1.0, 1.0),))),
        ),
        ((0, 0), (0, 1)),
        ((3, 0),),
    )
    if family == "converted":
        converter = p.CategoricalConverter(
            "kind", 0, 1, ("low", "high"), "neighbor_disc", ((-2.0,), (3.0,))
        )
        base = p.ConvertedCausalRecipe(
            base.task,
            tuple(
                p.ConvertedCausalNode(node, (converter,) if i == 2 else ())
                for i, node in enumerate(base.nodes)
            ),
            (p.ConvertedReadout(2, capture="kind"), p.ConvertedReadout(2, channel=1)),
            p.ConvertedReadout(3, channel=0),
        )
    elif family == "native":
        base = p.NativeCausalRecipe(
            base.task, tuple(p.NativeCausalNode(n) for n in base.nodes), ((2, 0), (2, 1)), (3, 0)
        )
    elif family == "vector":
        base = p.NativeVectorRecipe(
            base.task,
            tuple(
                p.NativeVectorNode(n, (p.VariableDomain(),) * n.function.output_width)
                for n in base.nodes
            ),
            ((2, 0), (2, 1)),
            (3, 0),
        )
    plan = p.RecursiveForestPlan(
        estimators=1, min_depth=1, max_depth=1, lifecycle="fresh-forest-per-row", populations=16
    )
    recipe = p.realize_recursive_population_graph(
        p.RecursivePopulationGraphRequest(base, (p.RecursiveNodePopulationPlan(3, plan),)),
        seed=17,
        task_id=19,
    )
    task = p.generate_task(recipe, seed=17, task_id=19)
    rows = [13, 1, 13]
    graph = p.explain_generation(task, rows)
    nodes = {n["id"]: n for n in graph["nodes"]}
    incoming = np.asarray(nodes["dag-2-inputs"]["values"])
    left, right = incoming[:, :2], incoming[:, 2:]
    kernel, bias = conv.function.layers[0].kernel[0][0][0], conv.function.layers[0].bias[0]
    expected_conv = kernel * left + bias
    expected_network = right * (2.0, -1.0) + (1.0, 0.5)
    expected_product = left * right
    np.testing.assert_allclose(
        nodes["dag-2-detail-parent-0-layer-0-affine"]["values"], expected_conv
    )
    np.testing.assert_allclose(
        nodes["dag-2-detail-parent-1-block-1-expert-0-affine"]["values"], expected_network
    )
    np.testing.assert_allclose(nodes["dag-2-detail-joint-product"]["values"], expected_product)
    np.testing.assert_allclose(
        nodes["dag-2-detail-reduction"]["values"],
        (expected_conv + expected_network + expected_product) / 3,
    )
    shared = [n for n in graph["nodes"] if "construction-draw" in n["id"]]
    assert shared
    assert all(
        n["shape"][0] == 1 and "not an emitted dataset row" in n["parameters"]["scope"]
        for n in shared
    )
    assert all("original_rows" not in n["parameters"] for n in shared)
    seen: set[str] = set()
    for node in graph["nodes"]:
        assert set(node["parents"]) <= seen
        assert node["id"] not in seen
        seen.add(node["id"])


@pytest.mark.parametrize("block", [False, True])
def test_direct_network_details_close_at_original_rows(block: bool):
    network = p.FrozenNetworkFunction(
        2, (p.NetworkBlock((0,), (p.NetworkAffine(((2.0, -3.0),), (1.0,)),)),)
    )
    if block:
        network = p.FrozenNetworkBlockFunction((2,), network.blocks[0])
    base = p.FunctionalCausalRecipe(
        p.LinearRecipe(rows=16, context_rows=7, features=1, signal_features=1, active_features=1),
        (
            p.FunctionalCausalNode((), network, p.JointRootBase("gaussian", 2)),
            p.FunctionalCausalNode((0,), p.FrozenLinearFunction(((1.0,),))),
        ),
        ((0, 0),),
        ((1, 0),),
    )
    plan = p.RecursiveForestPlan(
        input_width=1,
        intervals=((-1.0, 1.0),),
        estimators=1,
        min_depth=1,
        max_depth=1,
        lifecycle="fresh-forest-per-row",
        populations=16,
    )
    recipe = p.realize_recursive_population_graph(
        p.RecursivePopulationGraphRequest(base, (p.RecursiveNodePopulationPlan(1, plan),)),
        seed=31,
        task_id=37,
    )
    task = p.generate_task(recipe, seed=31, task_id=37)
    nodes = {n["id"]: n for n in p.explain_generation(task, [9, 0, 9])["nodes"]}
    values = np.asarray(nodes["dag-0-inputs"]["values"])
    np.testing.assert_allclose(
        np.asarray(nodes["dag-0-detail-block-1-expert-0-affine"]["values"])[:, 0],
        2 * values[:, 0] - 3 * values[:, 1] + 1,
    )
    assert nodes["dag-0-function"]["parents"] == ["dag-0-detail-block-1-state"]


def test_nested_mlp_product_and_parent_reducer_expose_hidden_stages():
    activation = p.CalibratedActivation(
        kind="tanh",
        input_mean=(0.0,),
        input_inverse_std=(1.0,),
        shift=(0.0,),
        output_mean=(0.0,),
        output_inverse_std=(1.0,),
        shift_rows=(0,),
        calibration_rows=1,
        calibration_id="authored-tanh",
        calibration_digest="0" * 64,
    )
    mlp = p.FrozenMLPFunction(
        (p.FrozenLinearFunction(((2.0,),)),),
        (activation,),
        start_activation=False,
        end_activation=True,
    )
    product = p.FrozenKumoProductFunction(mlp, p.FrozenLinearFunction(((3.0,),)))
    reducer = p.FrozenMLPFunction(
        (p.FrozenLinearFunction(((1.0, -2.0),)),),
        (activation,),
        start_activation=False,
        end_activation=True,
    )
    root = p.FunctionalCausalNode(
        (), p.FrozenLinearFunction(((1.0,),)), p.JointRootBase("gaussian", 1)
    )
    base = p.FunctionalCausalRecipe(
        p.LinearRecipe(rows=16, context_rows=7, features=1, signal_features=1, active_features=1),
        (
            root,
            root,
            p.FunctionalCausalNode(
                (0, 1),
                p.FrozenParentFunction(
                    (product, p.FrozenLinearFunction(((4.0,),))), aggregation="mlp", reducer=reducer
                ),
            ),
            p.FunctionalCausalNode((2,), p.FrozenLinearFunction(((1.0,),))),
        ),
        ((2, 0),),
        ((3, 0),),
    )
    plan = p.RecursiveForestPlan(
        input_width=1,
        intervals=((-1.0, 1.0),),
        estimators=1,
        min_depth=1,
        max_depth=1,
        lifecycle="fresh-forest-per-row",
        populations=16,
    )
    recipe = p.realize_recursive_population_graph(
        p.RecursivePopulationGraphRequest(base, (p.RecursiveNodePopulationPlan(3, plan),)),
        seed=11,
        task_id=17,
    )
    graph = p.explain_generation(p.generate_task(recipe, seed=11, task_id=17), [15, 2, 15])
    nodes = {n["id"]: n for n in graph["nodes"]}
    x = np.asarray(nodes["dag-2-inputs"]["values"])
    np.testing.assert_allclose(
        np.asarray(nodes["dag-2-detail-parent-0-left-activation-0"]["values"])[:, 0],
        np.tanh(2 * x[:, 0]),
    )
    expected = np.tanh(3 * x[:, 0] * np.tanh(2 * x[:, 0]) - 8 * x[:, 1])
    np.testing.assert_allclose(
        np.asarray(nodes["dag-2-detail-reducer-activation-0"]["values"])[:, 0], expected
    )
    seen: set[str] = set()
    for n in graph["nodes"]:
        assert n["id"] not in seen
        assert set(n["parents"]) <= seen
        seen.add(n["id"])
