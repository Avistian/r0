"""Independent row-owned tree paths through multiple internal nodes and descendants."""

import math
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from numpy.typing import NDArray

from r0.modules import priors as p


def graph_request(family: str = "functional") -> p.RecursivePopulationGraphRequest:
    controls = p.LinearRecipe(
        rows=16, context_rows=8, features=2, signal_features=2, active_features=2
    )
    nodes = (
        p.FunctionalCausalNode(
            (), p.FrozenLinearFunction(((1.0, 0.0), (0.0, 1.0))), p.JointRootBase("gaussian", 2)
        ),
        p.FunctionalCausalNode((0,), p.FrozenLinearFunction(((1.0, 0.0),))),
        p.FunctionalCausalNode((0, 1), p.FrozenLinearFunction(((1.0, 2.0, 3.0),))),
        p.FunctionalCausalNode((1, 2), p.FrozenLinearFunction(((3.0, -2.0),))),
    )
    if family == "native":
        base = p.NativeCausalRecipe(
            controls, tuple(p.NativeCausalNode(n) for n in nodes), ((0, 0), (0, 1)), (3, 0)
        )
    elif family == "vector":
        base = p.NativeVectorRecipe(
            controls,
            tuple(
                p.NativeVectorNode(n, (p.VariableDomain(),) * n.function.output_width)
                for n in nodes
            ),
            ((0, 0), (0, 1)),
            (3, 0),
        )
    elif family == "converted":
        base = p.ConvertedCausalRecipe(
            controls,
            tuple(p.ConvertedCausalNode(n) for n in nodes),
            (p.ConvertedReadout(0, channel=0), p.ConvertedReadout(0, channel=1)),
            p.ConvertedReadout(3, channel=0),
        )
    else:
        base = p.FunctionalCausalRecipe(controls, nodes, ((0, 0), (0, 1)), ((3, 0),))
    first = p.RecursiveForestPlan(
        input_width=2,
        estimators=1,
        min_depth=1,
        max_depth=1,
        lifecycle="fresh-forest-per-row",
        populations=16,
    )
    second = replace(first, input_width=3, intervals=((-3.0, 3.0),) * 3)
    return p.RecursivePopulationGraphRequest(
        base, (p.RecursiveNodePopulationPlan(1, first), p.RecursiveNodePopulationPlan(2, second))
    )


def literal(sample: p.SampledRecursiveForest, inputs: NDArray[Any]) -> NDArray[Any]:
    outputs: list[list[float]] = []
    for forest, row in zip(sample.forests, inputs, strict=True):
        leaves: list[p.RecursiveTreeNode] = []
        for tree in forest.trees:
            index = 0
            while tree.nodes[index].left >= 0:
                node = tree.nodes[index]
                index = node.left if row[node.feature] <= node.threshold else node.right
            leaves.append(tree.nodes[index])
        outputs.append(
            [
                math.fsum(n.value[j] / len(leaves) for n in leaves)
                for j in range(forest.output_width)
            ]
        )
    return np.asarray(outputs)


@pytest.mark.parametrize("family", ["functional", "native", "vector", "converted"])
def test_two_row_owned_nodes_and_downstream_literal_arithmetic(family: str):
    request = graph_request(family)
    recipe = p.realize_recursive_population_graph(request, seed=23, task_id=2**54 + 9)
    task = p.generate_task(recipe, seed=23, task_id=2**54 + 9)
    first, second = recipe.bindings
    np.testing.assert_array_equal(first.inputs, task.inputs)
    first_values = literal(first.sample, task.inputs.numpy())
    second_inputs = np.column_stack([task.inputs.numpy(), first_values])
    np.testing.assert_array_equal(second.inputs, second_inputs)
    second_values = literal(second.sample, second_inputs)
    np.testing.assert_allclose(
        task.targets.numpy(), 3 * first_values[:, 0] - 2 * second_values[:, 0], rtol=0, atol=1e-14
    )
    assert first.sample.final == second.sample.initial


@pytest.mark.parametrize("family", ["functional", "native", "vector", "converted"])
def test_complete_graph_archive_roundtrip(family: str):
    import json
    from dataclasses import asdict

    recipe = p.realize_recursive_population_graph(graph_request(family), seed=23, task_id=2**54 + 9)
    task = p.generate_task(recipe, seed=23, task_id=2**54 + 9)
    decoded = p.recipe_from_dict(json.loads(json.dumps(asdict(recipe))))
    assert decoded == recipe
    replay = p.generate_task(decoded, seed=23, task_id=2**54 + 9)
    assert replay.recipe_json == task.recipe_json
    assert replay.mechanism_json == task.mechanism_json


@pytest.mark.parametrize("family", ["native", "vector"])
def test_internal_majority_labels_feed_one_hot_descendants(family: str):
    request = graph_request(family)
    controls = request.task
    root = p.FunctionalCausalNode(
        (), p.FrozenLinearFunction(((1.0, 0.0), (0.0, 1.0))), p.JointRootBase("gaussian", 2)
    )
    classifier = p.FunctionalCausalNode(
        (0,), p.FrozenLinearFunction(((1.0, 0.0), (0.0, 1.0), (-1.0, -1.0)))
    )
    child = p.FunctionalCausalNode((1,), p.FrozenLinearFunction(((1.0, 2.0, 4.0),)))
    domain = p.VariableDomain("categorical", ("a", "b", "c"), "argmax")
    if family == "native":
        base = p.NativeCausalRecipe(
            controls,
            (
                p.NativeCausalNode(root),
                p.NativeCausalNode(classifier, domain),
                p.NativeCausalNode(child),
            ),
            ((0, 0), (0, 1)),
            (2, 0),
        )
    else:
        base = p.NativeVectorRecipe(
            controls,
            (
                p.NativeVectorNode(root, (p.VariableDomain(), p.VariableDomain())),
                p.NativeVectorNode(classifier, (domain,)),
                p.NativeVectorNode(child, (p.VariableDomain(),)),
            ),
            ((0, 0), (0, 1)),
            (2, 0),
        )
    plan = p.RecursiveForestPlan(
        estimators=3,
        min_depth=1,
        max_depth=1,
        task="classification",
        leaf_mean=(),
        leaf_scale=(),
        classes=3,
        class_assignment="random-start-cycle",
        tie_break="first-tree",
        lifecycle="fresh-forest-per-row",
        populations=16,
    )
    recipe = p.realize_recursive_population_graph(
        p.RecursivePopulationGraphRequest(base, (p.RecursiveNodePopulationPlan(1, plan),)),
        seed=23,
        task_id=7,
    )
    task = p.generate_task(recipe, seed=23, task_id=7)
    expected: list[float] = []
    for forest, row in zip(recipe.bindings[0].sample.forests, task.inputs.numpy(), strict=True):
        votes: list[int] = []
        for tree in forest.trees:
            node = tree.nodes[0]
            value = tree.nodes[
                node.left if row[node.feature] <= node.threshold else node.right
            ].label
            assert value is not None
            votes.append(value)
        most = max(votes.count(v) for v in votes)
        label = next(v for v in votes if votes.count(v) == most)
        assert label is not None
        expected.append((1.0, 2.0, 4.0)[label])
    np.testing.assert_array_equal(task.targets, expected)


@pytest.mark.parametrize("mode", ["matched", "population"])
def test_internal_hard_clamp_and_soft_override(mode: str):
    request = graph_request()
    recipe = p.realize_recursive_population_graph(request, seed=23, task_id=7)
    change = p.CausalInterventionRecipe(
        request.base,
        (
            p.HardIntervention(1, (0,), (5.0,)),
            p.SoftIntervention(2, function=p.FrozenLinearFunction(((0.0, 0.0, 2.0),))),
        ),
        mode=mode,
        population=3,
    )
    intervened = p.intervene_recursive_population_graph(recipe, change)
    task = p.generate_task(intervened, seed=23, task_id=7)
    np.testing.assert_array_equal(task.targets, [-5.0] * 16)
    assert intervened.inputs[0] == (1, None)
    assert intervened.inputs[1][1] is not None
    np.testing.assert_array_equal(np.asarray(intervened.inputs[1][1])[:, 2], [5.0] * 16)
    assert (intervened.samples[0] == recipe.bindings[0].sample) == (mode == "matched")


@pytest.mark.parametrize("mode", ["matched", "population"])
def test_intervened_graph_archive(mode: str):
    import json
    from dataclasses import asdict

    base = p.realize_recursive_population_graph(graph_request(), seed=23, task_id=7)
    changed = p.intervene_recursive_population_graph(
        base,
        p.CausalInterventionRecipe(
            base.request.base, (p.HardIntervention(1, (0,), (3.0,)),), mode=mode
        ),
    )
    assert p.recipe_from_dict(json.loads(json.dumps(asdict(changed)))) == changed


@pytest.mark.parametrize("family", ["functional", "native", "vector", "converted"])
def test_original_rows_graph_trace(family: str):
    recipe = p.realize_recursive_population_graph(graph_request(family), seed=23, task_id=7)
    task = p.generate_task(recipe, seed=23, task_id=7)
    graph = p.explain_generation(task, indices=[15, 0, 15])
    nodes = {n["id"]: n for n in graph["nodes"]}
    assert graph["rows"] == [15, 0, 15]
    np.testing.assert_array_equal(
        np.asarray(nodes["target"]["values"])[:, 0], task.targets[[15, 0, 15]]
    )
    assert nodes["dag-3-inputs"]["parents"] == ["dag-1-state", "dag-2-state"]
    assert nodes["population-2-tree-0"]["parameters"]["original_rows"] == [15, 0, 15]
    seen: set[str] = set()
    for node in graph["nodes"]:
        assert set(node["parents"]) <= seen
        seen.add(node["id"])


@pytest.mark.parametrize("mode", ["matched", "population"])
def test_stochastic_target_trace_and_comparison(mode: str):
    import json
    from dataclasses import asdict

    request = graph_request()
    base = replace(request.base, task=replace(request.task, noise_ratio=0.7))
    recipe = p.realize_recursive_population_graph(replace(request, base=base), seed=23, task_id=7)
    change = p.CausalInterventionRecipe(
        base, (p.HardIntervention(1, (0,), (2.0,)),), mode=mode, population=4
    )
    pair = p.generate_population_graph_pair(recipe, change)
    assert pair.before.recipe_json == p.generate_task(recipe, seed=23, task_id=7).recipe_json
    card = json.loads(pair.card_json)
    assert card["base_archive_unchanged"]
    assert card["split_equal"]
    assert card["forests_equal"] == (mode == "matched")
    graph = p.explain_generation(pair.after, [15, 0, 15])
    nodes = {n["id"]: n for n in graph["nodes"]}
    assert nodes["target"]["parents"] == ["signal", "noise"]
    np.testing.assert_array_equal(
        np.asarray(nodes["target"]["values"])[:, 0], pair.after.targets[[15, 0, 15]]
    )
    assert "dag-1-function" not in nodes
    assert nodes["dag-1-state"]["parameters"]["interventions"]
    changed = p.intervene_recursive_population_graph(recipe, change)
    assert p.recipe_from_dict(asdict(changed)) == changed


@pytest.mark.parametrize(
    "engine",
    [np.random.PCG64, np.random.PCG64DXSM, np.random.Philox, np.random.SFC64, np.random.MT19937],
)
def test_node_major_construction_stream_is_owned_and_replayable(
    engine: type[np.random.BitGenerator],
):
    from dataclasses import asdict

    rng = np.random.Generator(engine(31))
    recipe = p.realize_recursive_population_graph(graph_request(), seed=23, task_id=7, rng=rng)
    assert recipe.bindings[0].sample.initial.engine == engine.__name__
    assert repr(rng.bit_generator.state) == repr(
        recipe.bindings[-1].sample.final.restore().bit_generator.state
    )
    assert recipe.bindings[0].sample.final == recipe.bindings[1].sample.initial
    assert p.recipe_from_dict(asdict(recipe)) == recipe


@pytest.mark.parametrize("family", ["functional", "native", "vector", "converted"])
def test_whole_graph_limit_rejects_before_any_receipt_restore(family: str):
    from dataclasses import asdict
    from unittest.mock import patch

    recipe = p.realize_recursive_population_graph(graph_request(family), seed=23, task_id=7)
    with (
        patch.object(
            type(recipe.bindings[0].sample.initial),
            "restore",
            side_effect=AssertionError("early restore"),
        ),
        pytest.raises(ValueError, match="population|recursive|limit|recipe|semantic|complete"),
    ):
        p.recipe_from_dict(asdict(recipe), limits=replace(recipe.limits, working_bytes=32 * 1024))


def test_tampered_row_binding_and_failed_realization_preserve_caller():
    from unittest.mock import patch

    recipe = p.realize_recursive_population_graph(graph_request(), seed=23, task_id=7)
    first = replace(recipe.bindings[0], inputs=tuple(reversed(recipe.bindings[0].inputs)))
    with pytest.raises(ValueError, match="input rows differ"):
        replace(recipe, bindings=(first, recipe.bindings[1]))
    rng = np.random.default_rng(13)
    initial = repr(rng.bit_generator.state)
    with (
        patch.object(
            p.RecursivePopulationGraph,
            "validate_replay",
            side_effect=ValueError("late graph rejection"),
        ),
        pytest.raises(ValueError, match="late graph rejection"),
    ):
        p.realize_recursive_population_graph(graph_request(), seed=23, task_id=7, rng=rng)
    assert repr(rng.bit_generator.state) == initial


def test_archive_complete_fields_and_task_semantics():
    import json
    from dataclasses import asdict

    recipe = p.realize_recursive_population_graph(graph_request(), seed=23, task_id=7)
    raw = asdict(recipe)
    for key in raw:
        with pytest.raises(ValueError, match="population|recursive|limit|recipe|semantic|complete"):
            p.recipe_from_dict({k: v for k, v in raw.items() if k != key})
    task = p.generate_task(recipe, seed=23, task_id=7)
    for changed in (replace(task, scores=task.scores.float()), replace(task, task_type="binary")):
        with pytest.raises(ValueError, match="population|recursive|limit|recipe|semantic|complete"):
            p.explain_generation(changed, [0])
    record = json.loads(task.recipe_json)
    record["execution"]["implementation"] = "unrelated-generator-v1"
    with pytest.raises(ValueError, match="population|recursive|limit|recipe|semantic|complete"):
        p.explain_generation(replace(task, recipe_json=json.dumps(record)), [0])


def test_graph_comparison_archive_preserves_large_identity_and_baseline(tmp_path: Path):
    from r0.modules.generation import load_group, save_group

    identity = 2**54 + 17
    recipe = p.realize_recursive_population_graph(graph_request(), seed=23, task_id=identity)
    pair = p.generate_population_graph_pair(
        recipe,
        p.CausalInterventionRecipe(recipe.request.base, (p.HardIntervention(1, (0,), (1.5,)),)),
    )
    path = tmp_path / "dag.zip"
    save_group(path, [(identity, pair.after)], max_bytes=10_000_000)
    restored = next(load_group(path))
    assert restored.recipe_json == pair.after.recipe_json
    assert restored.mechanism_json == pair.after.mechanism_json
    assert p.explain_generation(restored, [15, 0, 15])["target_error"] == 0.0


def test_intervention_archive_complete_identity_and_pre_restore_caps():
    from dataclasses import asdict
    from unittest.mock import patch

    recipe = p.realize_recursive_population_graph(graph_request(), seed=23, task_id=7)
    changed = p.intervene_recursive_population_graph(
        recipe,
        p.CausalInterventionRecipe(recipe.request.base, (p.HardIntervention(1, (0,), (3.0,)),)),
    )
    raw = asdict(changed)
    for key in raw:
        with pytest.raises(ValueError, match="population|recursive|limit|recipe|semantic|complete"):
            p.recipe_from_dict({k: v for k, v in raw.items() if k != key})
    with (
        patch.object(
            type(recipe.bindings[0].sample.initial),
            "restore",
            side_effect=AssertionError("early restore"),
        ),
        pytest.raises(ValueError, match="population|recursive|limit|recipe|semantic|complete"),
    ):
        p.recipe_from_dict(raw, limits=replace(recipe.limits, working_bytes=32 * 1024))
