"""Public graph integration checked against literal traversal and majority arithmetic."""

import copy
import json
import math
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any
from unittest.mock import patch

import numpy as np
import pytest
import torch

from r0.modules import priors as p
from r0.modules.generation import load_group, save_group


def sampled(classification: bool = False, *, width: int = 2) -> p.SampledRecursiveForest:
    plan = p.RecursiveForestPlan(
        input_width=width,
        estimators=3,
        min_depth=1,
        max_depth=2,
        intervals=((-3.0, 3.0),) * width,
        task="classification" if classification else "regression",
        leaf_mean=() if classification else (0.0,),
        leaf_scale=() if classification else (1.0,),
        classes=3 if classification else 0,
        class_assignment="random-start-cycle" if classification else "none",
        tie_break="first-tree" if classification else "none",
    )
    return p.sample_recursive_forest(plan, rng=np.random.default_rng(37))


def literal(forest: p.RecursiveForest, rows: np.ndarray) -> np.ndarray:
    outputs: list[list[float]] = []
    for row in rows:
        leaves: list[p.RecursiveTreeNode] = []
        for tree in forest.trees:
            index = 0
            while tree.nodes[index].left >= 0:
                node = tree.nodes[index]
                index = node.left if row[node.feature] <= node.threshold else node.right
            leaves.append(tree.nodes[index])
        if forest.task == "regression":
            outputs.append(
                [
                    math.fsum(n.value[j] / len(leaves) for n in leaves)
                    for j in range(forest.output_width)
                ]
            )
        else:
            labels = [n.label for n in leaves]
            counts = {k: labels.count(k) for k in range(forest.classes)}
            tied = [k for k, count in counts.items() if count == max(counts.values())]
            winner = (
                next(k for k in labels if k in tied)
                if forest.tie_break == "first-tree"
                else min(tied)
                if forest.tie_break == "smallest"
                else max(tied)
            )
            outputs.append([float(k == winner) for k in range(forest.classes)])
    return np.asarray(outputs)


def native(classification: bool = False) -> p.NativeCausalRecipe:
    function = p.FrozenRecursiveForestFunction(sampled(classification))
    task = p.LinearRecipe(
        rows=16,
        context_rows=8,
        features=2,
        signal_features=2,
        active_features=2,
        task_type="multiclass" if classification else "regression",
        classes=3 if classification else 0,
    )
    root = p.NativeCausalNode(
        p.FunctionalCausalNode(
            (), p.FrozenLinearFunction(((1.0, 0.0), (0.0, 1.0))), p.JointRootBase("gaussian", 2)
        )
    )
    node = p.recursive_forest_native_node(
        function, parents=(0,), vocabulary=("a", "b", "c") if classification else ()
    )
    return p.NativeCausalRecipe(task, (root, node), ((0, 0), (0, 1)), (1, 0))


@pytest.mark.parametrize("classification", [False, True])
def test_literal_paths_votes_and_native_direct_task(classification: bool):
    recipe = native(classification)
    function = recipe.nodes[1].equation.function
    assert isinstance(function, p.FrozenRecursiveForestFunction)
    task = p.generate_task(recipe, seed=13, task_id=2**54 + 7)
    expected = literal(function.forest, task.inputs.numpy())
    torch.testing.assert_close(
        p.evaluate_frozen_function(function, task.inputs), torch.tensor(expected), rtol=0, atol=0
    )
    target = expected.argmax(1) if classification else expected[:, 0]
    np.testing.assert_array_equal(task.targets.numpy(), target)
    assert task.target_origin == "native-variable-v1"
    assert task.targets.dtype == (torch.int64 if classification else torch.float64)
    assert task.classes == (("a", "b", "c") if classification else ())
    graph = p.explain_generation(task, [15, 0, 15])
    assert graph["rows"] == [15, 0, 15]
    by_id = {n["id"]: n for n in graph["nodes"]}
    assert by_id["function-1"]["parents"] == ["recursive-1-aggregate"]
    assert (
        by_id["recursive-1-aggregate"]["parameters"]["output_semantics"]
        == function.output_semantics
    )
    assert by_id["recursive-1-tree-0"]["parameters"]["original_rows"] == [15, 0, 15]
    assert by_id["targets"]["values"] == [[v] for v in target[[15, 0, 15]].tolist()]
    # Full request/receipt is inside the versioned function envelope.
    raw = json.loads(task.recipe_json)["recipe"]
    assert raw["nodes"][1]["equation"]["function"]["source"] == json.loads(
        json.dumps(asdict(function.source))
    )
    reopened = p.recipe_from_dict(raw)
    replay = p.generate_task(reopened, seed=13, task_id=2**54 + 7)
    assert replay.recipe_json == task.recipe_json
    torch.testing.assert_close(replay.targets, task.targets, rtol=0, atol=0)


@pytest.mark.parametrize("classification", [False, True])
def test_exact_identity_archive_and_matched_interventions(tmp_path: Path, classification: bool):
    recipe = native(classification)
    identity = 2**55 + 19
    task = p.generate_task(recipe, seed=31, task_id=identity)
    path = tmp_path / "forest.zip"
    save_group(path, [(identity, task)], max_bytes=10_000_000)
    loaded = next(load_group(path))
    assert loaded.recipe_json == task.recipe_json
    assert loaded.episode() == task.episode()
    change = p.HardIntervention(0, (0, 1), (0.25, -0.75))
    pair = p.generate_causal_pair(recipe, interventions=(change,), seed=31, task_id=identity)
    function = recipe.nodes[1].equation.function
    assert isinstance(function, p.FrozenRecursiveForestFunction)
    expect = literal(function.forest, np.tile([0.25, -0.75], (16, 1)))
    np.testing.assert_array_equal(
        pair.after.targets.numpy(), expect.argmax(1) if classification else expect[:, 0]
    )
    assert json.loads(pair.after.recipe_json)["recipe"]["base"] == json.loads(
        json.dumps(asdict(recipe))
    )
    p.explain_generation(pair.after, [9, 0, 9])
    if classification:
        forced = p.generate_causal_pair(
            recipe, interventions=(p.HardIntervention(1, category="c"),), seed=31, task_id=identity
        )
        assert torch.all(forced.after.targets == 2)


def test_functional_parent_composition_and_trace():
    source = sampled(width=1)
    function = p.FrozenRecursiveForestFunction(source)
    parent = p.FrozenParentFunction((function, function), "mean")
    root = p.FunctionalCausalNode(
        (), p.FrozenLinearFunction(((1.0,),)), p.JointRootBase("gaussian", 1)
    )
    recipe = p.FunctionalCausalRecipe(
        p.LinearRecipe(rows=16, context_rows=8, features=2, signal_features=2, active_features=2),
        (root, root, p.FunctionalCausalNode((0, 1), parent)),
        ((0, 0), (1, 0)),
        ((2, 0),),
    )
    task = p.generate_task(recipe, seed=17, task_id=8)
    expected = (
        literal(function.forest, task.inputs[:, :1].numpy())
        + literal(function.forest, task.inputs[:, 1:].numpy())
    ) / 2
    np.testing.assert_allclose(task.targets.numpy(), expected[:, 0], rtol=0, atol=1e-15)
    graph = p.explain_generation(task, [4, 1, 4])
    by_id = {n["id"]: n for n in graph["nodes"]}
    assert by_id["parent-member-2-0"]["parents"] == ["parent-member-2-0-aggregate"]
    assert p.recipe_from_dict(asdict(recipe)) == recipe


def test_per_row_not_frozen_and_strict_custody_limits():
    source = sampled()
    per_row = p.sample_recursive_forest(
        replace(source.plan, lifecycle="fresh-forest-per-row", populations=2),
        rng=np.random.default_rng(2),
    )
    with pytest.raises(ValueError, match="per-row"):
        p.FrozenRecursiveForestFunction(per_row)
    function = p.FrozenRecursiveForestFunction(source)
    raw: dict[str, Any] = asdict(function)
    variants: list[dict[str, Any]] = []
    for key in raw:
        bad = copy.deepcopy(raw)
        del bad[key]
        variants.append(bad)
    bad = copy.deepcopy(raw)
    bad["source"]["raw"][0]["values"] = (999,)
    variants.append(bad)
    for bad in variants:
        with pytest.raises(ValueError, match="recursive|function|archive"):
            p.frozen_function_from_dict(bad)
    with (
        patch("copy.deepcopy", side_effect=AssertionError("snapshot before caller admission")),
        pytest.raises(ValueError, match="limit|admission|geometry"),
    ):
        p.frozen_function_from_dict(raw, limits=p.CausalLimits(nodes=1))
    with (
        patch("torch.Tensor.numpy", side_effect=AssertionError("allocation before admission")),
        pytest.raises(ValueError, match="limit|workspace"),
    ):
        p.evaluate_frozen_function(
            function, torch.zeros((16, 2)), limits=p.CausalLimits(working_bytes=1)
        )
    bad_recipe = asdict(native())
    bad_recipe["nodes"][1]["equation"]["function"]["source"]["raw"][0]["values"] = (999,)
    with pytest.raises(ValueError, match="recursive|archive"):
        p.recipe_from_dict(bad_recipe)


@pytest.mark.parametrize("tie", ["smallest", "largest", "first-tree"])
def test_authored_majority_ties_and_float64_threshold(tie: str):
    threshold = float(1 + 2**-30)
    trees = tuple(
        p.RecursiveTree(
            (
                p.RecursiveTreeNode(0, threshold, (0.0, 2.0), 0, 1, 2),
                p.RecursiveTreeNode(0, 0.0, (0.0, threshold), 1, label=left),
                p.RecursiveTreeNode(0, 2.0, (threshold, 2.0), 1, label=right),
            )
        )
        for left, right in ((2, 1), (1, 2))
    )
    forest = p.RecursiveForest(1, trees, "classification", classes=3, tie_break=tie)
    function = p.FrozenRecursiveForestFunction(forest)
    x = torch.tensor([[threshold], [np.nextafter(threshold, math.inf)], [0.0]], dtype=torch.float64)
    expected = literal(forest, x.numpy())
    np.testing.assert_array_equal(p.evaluate_frozen_function(function, x).numpy(), expected)
    assert p.frozen_function_from_dict(asdict(function)) == function
    if tie == "first-tree":
        assert expected.argmax(1).tolist() == [2, 1, 2]
    for order in ([2, 0, 2], [1, 0]):
        np.testing.assert_array_equal(
            p.evaluate_frozen_function(function, x[order]).numpy(), expected[order]
        )


def test_native_root_paths_and_encoded_class_descendant():
    source = sampled(True)
    function = p.FrozenRecursiveForestFunction(source)
    root = p.recursive_forest_native_node(
        function, root=p.JointRootBase("gaussian", 2), vocabulary=("a", "b", "c")
    )
    child = p.NativeCausalNode(
        p.FunctionalCausalNode((0,), p.FrozenLinearFunction(((2.0, 5.0, 11.0),)))
    )
    task_control = p.LinearRecipe(
        rows=16, context_rows=8, features=1, signal_features=1, active_features=1
    )
    recipe = p.NativeCausalRecipe(task_control, (root, child), ((0, 0),), (1, 0))
    task = p.generate_task(recipe, seed=7, task_id=3)
    np.testing.assert_array_equal(
        task.targets.numpy(), np.array([2.0, 5.0, 11.0])[task.inputs[:, 0].numpy().astype(int)]
    )
    graph = p.explain_generation(task, [15, 0, 15])
    nodes = {n["id"]: n for n in graph["nodes"]}
    assert nodes["recursive-0-tree-0"]["parents"] == ["root-0"]
    root_values = np.array(nodes["root-0"]["values"])
    np.testing.assert_array_equal(
        nodes["recursive-0-aggregate"]["values"], literal(function.forest, root_values)
    )
    card = json.loads(task.mechanism_json)
    assert card["recursive_forests"][0]["construction"] == "sampled-with-complete-raw-receipt"
    assert "conditional" in card["recursive_forests"][0]["moment_scope"]


def vector_function() -> p.FrozenRecursiveForestFunction:
    plan = replace(
        sampled(width=1).plan, source="r0-extension-v1", leaf_mean=(0.0, 1.0), leaf_scale=(1.0, 0.5)
    )
    return p.FrozenRecursiveForestFunction(
        p.sample_recursive_forest(plan, rng=np.random.default_rng(21))
    )


def test_vector_joint_and_converted_paths_preserve_tree_outputs():
    function = vector_function()
    equation = p.FunctionalCausalNode((), function, p.JointRootBase("gaussian", 1))
    controls = p.LinearRecipe(
        rows=16, context_rows=8, features=1, signal_features=1, active_features=1
    )
    vector = p.NativeVectorRecipe(
        controls,
        (p.NativeVectorNode(equation, (p.VariableDomain(), p.VariableDomain())),),
        ((0, 0),),
        (0, 1),
    )
    task = p.generate_task(vector, seed=19, task_id=11)
    trace = p.explain_generation(task, [15, 0, 15])
    nodes = {n["id"]: n for n in trace["nodes"]}
    expected = literal(function.forest, np.array(nodes["root-0"]["values"]))
    np.testing.assert_array_equal(nodes["recursive-0-aggregate"]["values"], expected)
    np.testing.assert_array_equal(task.targets[[15, 0, 15]].numpy(), expected[:, 1])
    assert p.recipe_from_dict(asdict(vector)) == vector
    converter = p.CategoricalConverter(
        "side", 0, 1, ("negative", "positive"), "neighbor_id", ((-1.0,), (1.0,))
    )
    converted = p.ConvertedCausalRecipe(
        controls,
        (p.ConvertedCausalNode(equation, (converter,)),),
        (p.ConvertedReadout(0, capture="side"),),
        p.ConvertedReadout(0, channel=1),
    )
    converted_task = p.generate_task(converted, seed=19, task_id=11)
    converted_trace = p.explain_generation(converted_task, [15, 0, 15])
    nodes = {n["id"]: n for n in converted_trace["nodes"]}
    expected = literal(function.forest, np.array(nodes["root-0"]["values"]))
    np.testing.assert_array_equal(converted_task.targets[[15, 0, 15]].numpy(), expected[:, 1])
    assert p.recipe_from_dict(asdict(converted)) == converted


def test_aggregate_replay_admission_precedes_all_rng_snapshots():
    function = p.FrozenRecursiveForestFunction(sampled(width=1))
    report = function.admit(16, p.CausalLimits())
    cap = report["estimated_working_bytes"] + 500_000
    # One retained construction is admitted, two simultaneous replay budgets are not.
    limits = p.CausalLimits(working_bytes=cap)
    p.evaluate_frozen_function(function, torch.zeros((16, 1)), limits=limits)
    parent = p.FrozenParentFunction((function, function), "mean")
    raw = asdict(parent)
    with (
        patch("copy.deepcopy", side_effect=AssertionError("RNG cloned before aggregate admission")),
        pytest.raises(ValueError, match="aggregate"),
    ):
        p.frozen_function_from_dict(raw, limits=limits)
    with (
        patch(
            "torch.Tensor.numpy",
            side_effect=AssertionError("values allocated before aggregate admission"),
        ),
        pytest.raises(ValueError, match="aggregate"),
    ):
        p.evaluate_frozen_function(parent, torch.zeros((16, 2)), limits=limits)


def test_class_indicator_width_is_admitted_before_raw_replay():
    function = p.FrozenRecursiveForestFunction(sampled(True))
    raw = asdict(function)
    with (
        patch(
            "copy.deepcopy",
            side_effect=AssertionError("RNG clone before indicator width admission"),
        ),
        pytest.raises(ValueError, match="class width"),
    ):
        p.frozen_function_from_dict(raw, limits=p.CausalLimits(node_channels=2))


def test_next_task_changes_inputs_without_resampling_construction():
    recipe = native(True)
    first = p.generate_task(recipe, seed=1, task_id=2**54 + 1)
    second = p.generate_task(recipe, seed=2, task_id=2**54 + 2)
    assert not torch.equal(first.inputs, second.inputs)
    assert json.loads(first.recipe_json)["recipe"] == json.loads(second.recipe_json)["recipe"]
    source = recipe.nodes[1].equation.function
    assert isinstance(source, p.FrozenRecursiveForestFunction)
    np.testing.assert_array_equal(
        second.targets.numpy(), literal(source.forest, second.inputs.numpy()).argmax(1)
    )


def test_function_stage_change_preserves_base_and_traces_effective_forest():
    recipe = native(True)
    source = sampled(True)
    replacement = p.FrozenRecursiveForestFunction(
        p.sample_recursive_forest(source.plan, rng=np.random.default_rng(99))
    )
    change = p.SoftIntervention(1, function=replacement)
    pair = p.generate_causal_pair(recipe, interventions=(change,), seed=17, task_id=7)
    np.testing.assert_array_equal(
        pair.after.targets.numpy(), literal(replacement.forest, pair.after.inputs.numpy()).argmax(1)
    )
    trace = p.explain_generation(pair.after, [15, 0, 15])
    seen: set[str] = set()
    for node in trace["nodes"]:
        assert set(node["parents"]) <= seen
        seen.add(node["id"])
    function = next(n for n in trace["nodes"] if n["id"] == "intervention-1-function")
    assert function["parents"] == ["intervention-recursive-1-aggregate"]
    np.testing.assert_array_equal(
        function["values"], literal(replacement.forest, pair.after.inputs[[15, 0, 15]].numpy())
    )
    decoded = p.recipe_from_dict(json.loads(pair.after.recipe_json)["recipe"])
    assert isinstance(decoded, p.CausalInterventionRecipe)
    assert decoded.base == recipe


def test_binary_labels_and_population_intervention_keep_frozen_receipt():
    original = native(True)
    plan = replace(sampled(True).plan, classes=2)
    function = p.FrozenRecursiveForestFunction(
        p.sample_recursive_forest(plan, rng=np.random.default_rng(5))
    )
    recipe = replace(
        original,
        task=replace(original.task, task_type="binary", classes=2),
        nodes=(
            original.nodes[0],
            p.recursive_forest_native_node(function, parents=(0,), vocabulary=("no", "yes")),
        ),
    )
    pair = p.generate_causal_pair(
        recipe,
        interventions=(p.HardIntervention(0, (0,), (0.5,)),),
        seed=5,
        task_id=7,
        mode="population",
        population=3,
    )
    assert pair.after.targets.dtype == torch.int64
    assert pair.after.classes == ("no", "yes")
    np.testing.assert_array_equal(
        pair.after.targets.numpy(), literal(function.forest, pair.after.inputs.numpy()).argmax(1)
    )
    archive = json.loads(pair.after.recipe_json)["recipe"]
    assert archive["base"] == json.loads(json.dumps(asdict(recipe)))
    reopened = p.generate_task(p.recipe_from_dict(archive), seed=5, task_id=7)
    torch.testing.assert_close(reopened.targets, pair.after.targets, rtol=0, atol=0)
