"""Category extraction and width-preserving state have distinct public meanings."""

import json
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import pytest
import torch

from r0.modules import priors as p


@pytest.mark.parametrize("mode", ["neighbor_id", "neighbor_disc", "neighbor_func", "neighbor_int"])
def test_neighbor_literal_distance_and_distinct_descendant_representations(mode: str) -> None:
    centers = ((-2.0, 1.0), (1.0, -1.0), (3.0, 2.0))
    values = np.asarray([[-1.0, 2.0], [0.5, -0.7], [2.8, 2.1], [0.0, 0.0]])
    converter = p.CategoricalConverter(
        "kind",
        1,
        3,
        ("a", "b", "c"),
        mode,
        representatives=centers,
        exponent=0.75,
        post_function=p.FrozenLinearFunction(((2.0, 1.0), (-1.0, 3.0)))
        if mode == "neighbor_func"
        else None,
    )
    emission = converter.apply(torch.tensor(values))
    expected = (abs(values[:, None] - np.asarray(centers)[None]) ** 0.75).sum(axis=2).argmin(axis=1)
    np.testing.assert_array_equal(emission.categories.numpy(), expected)
    np.testing.assert_array_equal(emission.probabilities.numpy(), np.eye(3)[expected])
    state = values if mode.endswith("_id") else np.asarray(centers)[expected]
    if mode.endswith("_int"):
        state = np.repeat(expected[:, None], 2, axis=1)
    if mode.endswith("_func"):
        state = state @ np.asarray([[2.0, 1.0], [-1.0, 3.0]]).T
    np.testing.assert_allclose(emission.state.numpy(), state, rtol=0, atol=1e-14)


@pytest.mark.parametrize("mode", ["softmax_id", "softmax_disc", "softmax_int"])
def test_softmax_literal_probabilities_uniforms_and_state(mode: str) -> None:
    values = np.asarray([[-1.0, 0.2, 3.0], [2.0, -0.5, 0.0], [0.2, 0.5, -0.8]])
    uniform = np.asarray([0.0, 0.6, 0.999])
    representatives = ((1.0, 2.0, 3.0), (-2.0, 0.0, 1.0), (0.0, 4.0, 5.0))
    converter = p.CategoricalConverter(
        "kind",
        0,
        3,
        ("a", "b", "c"),
        mode,
        representatives=representatives,
        mean=(0.5, -0.2, 0.0),
        scale=(2.0, 0.5, 1.0),
        bias=(-0.3, 0.7, 0.2),
        factor=0.8,
    )
    emission = converter.apply(torch.tensor(values), uniforms=torch.tensor(uniform))
    logits = 0.8 * (values - [0.5, -0.2, 0.0]) / [2.0, 0.5, 1.0] + [-0.3, 0.7, 0.2]
    probs = np.exp(logits - logits.max(axis=1, keepdims=True))
    probs /= probs.sum(axis=1, keepdims=True)
    expected = (uniform[:, None] >= probs.cumsum(axis=1)).sum(axis=1)
    np.testing.assert_allclose(emission.probabilities.numpy(), probs, rtol=1e-14)
    np.testing.assert_array_equal(emission.categories.numpy(), expected)
    state = values if mode.endswith("_id") else np.asarray(representatives)[expected]
    if mode.endswith("_int"):
        state = np.repeat(expected[:, None], 3, axis=1)
    np.testing.assert_allclose(emission.state.numpy(), state, rtol=0, atol=1e-14)


def test_category_capture_precedes_noise_and_importance_while_child_uses_representation() -> None:
    converter = p.CategoricalConverter(
        "kind", 0, 2, ("left", "right"), "neighbor_int", ((-1.0, -1.0), (1.0, 1.0))
    )
    root = p.ConvertedCausalNode(
        p.FunctionalCausalNode(
            (),
            p.FrozenLinearFunction(((1.0,), (1.0,), (1.0,))),
            p.JointRootBase("gaussian", 1),
            pipeline=p.CausalNodePipeline(3, node_importance=3.0),
        ),
        (converter,),
    )
    child = p.ConvertedCausalNode(
        p.FunctionalCausalNode((0,), p.FrozenLinearFunction(((2.0, 1.0, 0.0),)))
    )
    recipe = p.ConvertedCausalRecipe(
        p.LinearRecipe(rows=24, context_rows=12, features=2, signal_features=2, active_features=2),
        (root, child),
        (p.ConvertedReadout(0, capture="kind"), p.ConvertedReadout(0, channel=2)),
        p.ConvertedReadout(1, channel=0),
    )
    task = p.generate_task(recipe, seed=5, task_id=2)
    np.testing.assert_array_equal(
        task.inputs[:, 0].numpy(), (task.inputs[:, 1].numpy() > 0).astype(float)
    )
    np.testing.assert_array_equal(task.targets.numpy(), 9 * task.inputs[:, 0].numpy())
    assert task.feature_schema[0].kind == "binary"
    graph = p.explain_generation(task, [3, 1, 3])
    seen: set[str] = set()
    for node in graph["nodes"]:
        assert set(node["parents"]) <= seen, node["id"]
        seen.add(node["id"])
    assert "pipeline-0-category-kind-capture" in seen

    restored = p.generate_task(p.recipe_from_dict(asdict(recipe)), seed=5, task_id=2)
    assert torch.equal(restored.inputs, task.inputs)
    assert torch.equal(restored.targets, task.targets)


@pytest.mark.parametrize(
    "mode",
    [
        "neighbor_id",
        "neighbor_disc",
        "neighbor_func",
        "neighbor_int",
        "softmax_id",
        "softmax_disc",
        "softmax_int",
    ],
)
def test_sampled_fixed_converter_fit_and_ancestral_replay(mode: str, tmp_path: Path) -> None:
    functions = p.JointFunctionPrior(
        functions="lin", matrix=p.JointMatrixPrior(families=("gaussian",))
    )
    request = p.ConvertedGraphRequest(
        p.AttachmentGraphPrior(nodes=2),
        (("a", "b", "c"), ()),
        ("no", "yes"),
        rows=16,
        context_rows=8,
        calibration_rows=16,
        width_prior=p.JointWidthPrior(low=1, high=2),
        function_prior=functions,
        root_prior=p.JointRootPrior(families=("gaussian",), function=functions),
        converter_prior=p.CategoricalConverterPrior(
            modes=(mode,),
            function=functions,
            embeddings=p.JointRootPrior(families=("gaussian",), function=functions),
        ),
        pipeline_prior=p.JointPipelinePrior(
            channel_importance=False,
            l2_normalization=False,
            converter=p.JointNumericalConverterPrior(
                allow_kumaraswamy=False, allow_activation=False
            ),
        ),
    )
    graph = p.sample_converted_graph(
        request,
        **{
            name: np.random.default_rng(i)
            for i, name in enumerate(("named_rng", "raw_rng", "calibration_rng", "fit_rng"))
        },
    )
    assert len(graph.converters) == 2
    task = p.generate_task(graph, seed=3, task_id=9)
    assert task.targets.dtype == torch.int64
    assert task.classes == ("no", "yes")
    assert task.feature_schema[0].vocabulary == ("a", "b", "c")
    from r0.modules.generation import load_group, save_group

    save_group(tmp_path / "converted.zip", [(9, task)], max_bytes=20_000_000)
    assert next(load_group(tmp_path / "converted.zip")).episode() == task.episode()
    other = p.generate_task(
        graph.with_task(replace(graph.task, split=2, context_rows=5)), seed=3, task_id=9
    )
    assert other.source_task_id == task.source_task_id
    assert torch.equal(other.targets, task.targets)
    data = json.loads(json.dumps(asdict(graph)))
    data["converters"][0]["converter"]["representatives"][0][0] += 0.5
    with pytest.raises(ValueError, match="replay"):
        p.recipe_from_dict(data)
    with pytest.raises(ValueError, match="adapter"):
        p.paired_recipe(graph, intervention="noise", value=0.5)
    streams = {
        name: np.random.default_rng(i + 9)
        for i, name in enumerate(("named_rng", "raw_rng", "calibration_rng", "fit_rng"))
    }
    before = json.dumps([rng.bit_generator.state for rng in streams.values()])
    with pytest.raises(ValueError, match="limit|budget"):
        p.sample_converted_graph(replace(request, raw_limits=p.RawDrawLimits(draws=1)), **streams)
    assert json.dumps([rng.bit_generator.state for rng in streams.values()]) == before
    larger = p.generate_task(
        replace(graph.recipe, task=replace(graph.task, rows=32)), seed=3, task_id=9
    )
    assert torch.equal(larger.inputs[:16], task.inputs)
    assert torch.equal(larger.targets[:16], task.targets)
    graph.validate_replay()
    assert p.recipe_from_dict(asdict(graph)) == graph
    trace = p.explain_generation(task, [5, 2, 5])
    seen: set[str] = set()
    for node in trace["nodes"]:
        assert set(node["parents"]) <= seen, node["id"]
        seen.add(node["id"])


def test_captured_category_does_not_certify_unchanged_heavy_tailed_state() -> None:
    root = p.ConvertedCausalNode(
        p.FunctionalCausalNode(
            (),
            p.FrozenLinearFunction(((0.0,),)),
            p.JointRootBase("gaussian", 1),
            noise_scale=(1.0,),
            exogenous=p.ExogenousDistribution(p.InputDistribution("student_t", df=1.0)),
        )
    )
    category = p.CategoricalConverter(
        "side", 0, 1, ("negative", "positive"), "neighbor_disc", ((-1.0,), (1.0,))
    )
    middle = p.ConvertedCausalNode(
        p.FunctionalCausalNode((0,), p.FrozenLinearFunction(((1.0,),))), (category,)
    )
    standard = p.calibrate_standardization(
        np.arange(16, dtype=float)[:, None], calibration_id="independent-fixed"
    )
    final = p.ConvertedCausalNode(
        p.FunctionalCausalNode(
            (1,),
            p.FrozenLinearFunction(((1.0,),)),
            pipeline=p.CausalNodePipeline(1, stages=(standard,)),
        )
    )
    recipe = p.ConvertedCausalRecipe(
        p.LinearRecipe(rows=16, context_rows=8, features=1, active_features=1, signal_features=1),
        (root, middle, final),
        (p.ConvertedReadout(1, capture="side"),),
        p.ConvertedReadout(2, channel=0),
    )
    assert torch.isfinite(p.generate_task(recipe, seed=2, task_id=3).targets).all()
    with pytest.raises(ValueError, match="finite input second moments"):
        replace(
            recipe,
            nodes=(
                root,
                replace(middle, converters=(replace(category, mode="neighbor_id"),)),
                final,
            ),
        )


def test_converter_allocation_budget_and_malformed_payload_fail_closed() -> None:
    converter = p.CategoricalConverter(
        "kind",
        0,
        2,
        ("a", "b"),
        "softmax_disc",
        ((0.0, 1.0), (1.0, 0.0)),
        mean=(0.0, 0.0),
        scale=(1.0, 1.0),
        bias=(0.0, 0.0),
    )
    node = p.ConvertedCausalNode(
        p.FunctionalCausalNode(
            (), p.FrozenLinearFunction(((1.0,), (-1.0,), (1.0,))), p.JointRootBase("gaussian", 1)
        ),
        (converter,),
    )
    recipe = p.ConvertedCausalRecipe(
        p.LinearRecipe(rows=16, context_rows=8, features=1, signal_features=1, active_features=1),
        (node,),
        (p.ConvertedReadout(0, capture="kind"),),
        p.ConvertedReadout(0, channel=2),
    )
    with pytest.raises(ValueError, match="workspace|memory|storage"):
        replace(recipe, limits=replace(recipe.limits, working_bytes=1024))
    data = asdict(recipe)
    data["nodes"] = [{"equation": None, "converters": []}]
    with pytest.raises(ValueError, match="object"):
        p.recipe_from_dict(data)
    with pytest.raises(ValueError, match="uniform"):
        converter.apply(torch.ones((2, 2), dtype=torch.float64), uniforms=torch.tensor([0.0, 1.0]))
    with pytest.raises(ValueError, match="nonfinite"):
        replace(converter, factor=1e308).apply(
            torch.full((2, 2), 1e308, dtype=torch.float64), uniforms=torch.tensor([0.0, 0.5])
        )


@pytest.mark.parametrize("root_network", [True, False])
def test_converted_network_internals_use_original_root_or_parent_input(root_network: bool) -> None:
    network = p.FrozenNetworkFunction(
        1, (p.NetworkBlock((0,), (p.NetworkAffine(((2.0,), (-3.0,)), (0.5, -0.25)),)),)
    )
    converter = p.CategoricalConverter(
        "side", 0, 1, ("negative", "positive"), "neighbor_id", ((-1.0,), (1.0,))
    )
    node = p.ConvertedCausalNode(
        p.FunctionalCausalNode(
            () if root_network else (0,),
            network,
            p.JointRootBase("gaussian", 1) if root_network else None,
        ),
        (converter,),
    )
    nodes = (
        (node,)
        if root_network
        else (
            p.ConvertedCausalNode(
                p.FunctionalCausalNode(
                    (), p.FrozenLinearFunction(((1.0,),)), p.JointRootBase("gaussian", 1)
                )
            ),
            node,
        )
    )
    index = len(nodes) - 1
    recipe = p.ConvertedCausalRecipe(
        p.LinearRecipe(
            rows=16,
            context_rows=8,
            features=1,
            active_features=1,
            signal_features=1,
            task_type="binary",
            classes=2,
        ),
        nodes,
        (p.ConvertedReadout(index, channel=1),),
        p.ConvertedReadout(index, capture="side"),
    )
    task = p.generate_task(recipe, seed=41, task_id=9)
    trace = p.explain_generation(task, [15, 0, 15])
    by_id = {n["id"]: n for n in trace["nodes"]}
    original = np.asarray(by_id[f"network-{index}-input"]["values"])[:, 0]
    expected = np.stack((2 * original + 0.5, -3 * original - 0.25), axis=1)
    np.testing.assert_allclose(
        by_id[f"network-{index}-block-1-state"]["values"], expected, rtol=1e-14, atol=1e-14
    )
    np.testing.assert_array_equal(task.targets[[15, 0, 15]], expected[:, 0] > 0)
    seen: set[str] = set()
    for n in trace["nodes"]:
        assert set(n["parents"]) <= seen
        seen.add(n["id"])
