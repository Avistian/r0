"""Public original-row population custody and literal row-owned forest arithmetic."""

import json
import math
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch
from numpy.typing import NDArray

from r0.modules import priors as p


def request(classes: int = 0) -> p.RecursivePopulationRequest:
    task = p.LinearRecipe(rows=16, context_rows=8, features=2, signal_features=2, active_features=2)
    root = p.FunctionalCausalNode(
        (), p.FrozenLinearFunction(((1.0, 0.0), (0.0, 1.0))), p.JointRootBase("gaussian", 2)
    )
    child = p.FunctionalCausalNode((0,), p.FrozenLinearFunction(((1.0, 2.0),)))
    ancestors = p.FunctionalCausalRecipe(task, (root, child), ((0, 0), (0, 1)), ((1, 0),))
    plan = p.RecursiveForestPlan(
        input_width=2,
        estimators=2,
        min_depth=1,
        max_depth=1,
        task="classification" if classes else "regression",
        leaf_mean=() if classes else (0.0,),
        leaf_scale=() if classes else (1.0,),
        classes=classes,
        class_assignment="random-start-cycle" if classes else "none",
        tie_break="first-tree" if classes else "none",
        lifecycle="fresh-forest-per-row",
        populations=16,
    )
    return p.RecursivePopulationRequest(
        ancestors, ((1, 0), (0, 1)), plan, tuple(f"category-{i}" for i in range(classes))
    )


def literal(recipe: p.RecursivePopulationRecipe) -> NDArray[Any]:
    emitted: list[float | int | None] = []
    for forest, row in zip(recipe.sample.forests, recipe.parent_rows, strict=True):
        leaves: list[p.RecursiveTreeNode] = []
        for tree in forest.trees:
            index = 0
            while tree.nodes[index].left != -1:
                node = tree.nodes[index]
                index = node.left if row[node.feature] <= node.threshold else node.right
            leaves.append(tree.nodes[index])
        if forest.classes:
            labels = [n.label for n in leaves]
            best = max(labels.count(label) for label in labels)
            emitted.append(next(label for label in labels if labels.count(label) == best))
        else:
            emitted.append(math.fsum(n.value[0] / len(leaves) for n in leaves))
    return np.asarray(emitted)


@pytest.mark.parametrize("classes", [0, 2, 3])
def test_row_owned_arithmetic_and_exact_archive(classes: int):
    recipe = p.realize_recursive_population(request(classes), seed=13, task_id=2**54 + 7)
    task = p.generate_task(recipe, seed=13, task_id=2**54 + 7)
    np.testing.assert_array_equal(task.targets.numpy(), literal(recipe))
    np.testing.assert_array_equal(
        np.asarray(recipe.parent_rows)[:, 0], task.inputs[:, 0] + 2 * task.inputs[:, 1]
    )
    assert task.probabilities is None
    assert task.target_origin == "native-variable-v1"
    if classes:
        assert task.targets.dtype == torch.int64
    reopened = p.recipe_from_dict(json.loads(json.dumps(asdict(recipe))))
    replay = p.generate_task(reopened, seed=13, task_id=2**54 + 7)
    assert replay.recipe_json == task.recipe_json
    torch.testing.assert_close(replay.targets, task.targets, rtol=0, atol=0)
    with pytest.raises(ValueError, match="identity changed"):
        p.generate_task(recipe, seed=14, task_id=2**54 + 7)
    with pytest.raises(ValueError, match="population"):
        replace(recipe, seed=13.0)


@pytest.mark.parametrize("mode", ["matched", "population"])
def test_intervention_row_custody_trace_and_archive(mode: str):
    recipe = p.realize_recursive_population(request(), seed=13, task_id=7)
    changes = p.CausalInterventionRecipe(
        recipe.request.ancestors, (p.HardIntervention(0, channels=(0,), values=(7.0,)),)
    )
    after_recipe = p.intervene_recursive_population(recipe, ancestors=changes, mode=mode)
    after = p.generate_task(after_recipe, seed=13, task_id=7)
    torch.testing.assert_close(after.inputs[:, 0], torch.full((16,), 7.0, dtype=torch.float64))
    np.testing.assert_array_equal(
        np.asarray(after_recipe.parent_rows)[:, 0], 7 + 2 * after.inputs[:, 1]
    )
    assert (after_recipe.sample == recipe.sample) == (mode == "matched")
    decoded = p.recipe_from_dict(json.loads(json.dumps(asdict(after_recipe))))
    replay = p.generate_task(decoded, seed=13, task_id=7)
    assert replay.recipe_json == after.recipe_json
    graph = p.explain_generation(after, [15, 0, 15])
    assert graph["rows"] == [15, 0, 15]
    by_id = {node["id"]: node for node in graph["nodes"]}
    np.testing.assert_array_equal(
        np.asarray(by_id["targets"]["values"])[:, 0], after.targets[[15, 0, 15]]
    )
    assert by_id["population-tree-0"]["parameters"]["original_rows"] == [15, 0, 15]
    seen: set[str] = set()
    for node in graph["nodes"]:
        assert set(node["parents"]) <= seen
        seen.add(node["id"])


def test_forced_native_label_is_not_an_adapter_probability():
    recipe = p.realize_recursive_population(request(3), seed=13, task_id=7)
    changed = p.intervene_recursive_population(recipe, forced_category="category-2")
    task = p.generate_task(changed, seed=13, task_id=7)
    assert task.targets.numpy().tolist() == [2] * 16
    assert task.probabilities is None
    assert p.explain_generation(task, [15, 0, 15])["target_error"] == 0


@pytest.mark.parametrize("family", ["native", "vector", "converted"])
def test_all_ancestor_encoded_state_families(family: str):
    original = request()
    root, child = original.ancestors.nodes
    assert isinstance(root, p.FunctionalCausalNode)
    assert isinstance(child, p.FunctionalCausalNode)
    task = original.ancestors.task
    if family == "native":
        ancestors = p.NativeCausalRecipe(
            task, (p.NativeCausalNode(root), p.NativeCausalNode(child)), ((0, 0), (0, 1)), (1, 0)
        )
    elif family == "vector":
        ancestors = p.NativeVectorRecipe(
            task,
            (
                p.NativeVectorNode(root, (p.VariableDomain(), p.VariableDomain())),
                p.NativeVectorNode(child, (p.VariableDomain(),)),
            ),
            ((0, 0), (0, 1)),
            (1, 0),
        )
    else:
        ancestors = p.ConvertedCausalRecipe(
            task,
            (p.ConvertedCausalNode(root), p.ConvertedCausalNode(child)),
            (p.ConvertedReadout(0, channel=0), p.ConvertedReadout(0, channel=1)),
            p.ConvertedReadout(1, channel=0),
        )
    recipe = p.realize_recursive_population(
        replace(original, ancestors=ancestors), seed=5, task_id=6
    )
    task = p.generate_task(recipe, seed=5, task_id=6)
    np.testing.assert_array_equal(task.targets, literal(recipe))
    assert p.recipe_from_dict(asdict(recipe)) == recipe
    assert p.explain_generation(task, [15, 0, 15])["target_error"] == 0


def test_native_categorical_observation_and_encoded_parent_distinction():
    original = request(3)
    task = p.LinearRecipe(rows=16, context_rows=8, features=1, signal_features=1, active_features=1)
    root = p.NativeCausalNode(
        p.FunctionalCausalNode(
            (), p.FrozenLinearFunction(((1.0, 0.0), (0.0, 1.0))), p.JointRootBase("gaussian", 2)
        ),
        p.VariableDomain("categorical", ("red", "blue"), "softmax"),
    )
    child = p.NativeCausalNode(p.FunctionalCausalNode((0,), p.FrozenLinearFunction(((1.0, 2.0),))))
    ancestors = p.NativeCausalRecipe(task, (root, child), ((0, 0),), (1, 0))
    recipe = p.realize_recursive_population(
        replace(original, ancestors=ancestors, parents=((0, 0), (0, 1))), seed=5, task_id=6
    )
    result = p.generate_task(recipe, seed=5, task_id=6)
    np.testing.assert_array_equal(
        np.asarray(recipe.parent_rows), np.eye(2)[result.inputs[:, 0].numpy().astype(int)]
    )
    assert result.feature_schema[0].vocabulary == ("red", "blue")
    assert p.recipe_from_dict(asdict(recipe)) == recipe


@pytest.mark.parametrize(
    "engine",
    [np.random.PCG64, np.random.PCG64DXSM, np.random.Philox, np.random.SFC64, np.random.MT19937],
)
def test_injected_engine_custody_and_literal_row_ownership(engine: type[np.random.BitGenerator]):
    rng = np.random.Generator(engine(17))
    recipe = p.realize_recursive_population(request(), seed=5, task_id=7, rng=rng)
    assert recipe.sample.initial.engine == engine.__name__
    assert repr(rng.bit_generator.state) == repr(recipe.sample.final.restore().bit_generator.state)
    result = p.generate_task(recipe, seed=5, task_id=7)
    np.testing.assert_array_equal(result.targets, literal(recipe))
    assert p.recipe_from_dict(asdict(recipe)) == recipe


def test_explicit_pair_and_persisted_large_identity(tmp_path: Path):
    from r0.modules.generation import load_group, save_group

    recipe = p.realize_recursive_population(request(), seed=5, task_id=2**54 + 7)
    pair = p.generate_recursive_population_pair(recipe, forced_value=0.125)
    assert pair.before.recipe_json == p.generate_task(recipe, seed=5, task_id=2**54 + 7).recipe_json
    assert pair.after.targets.numpy().tolist() == [0.125] * 16
    graph = p.explain_generation(pair.after, [15, 0, 15])
    by_id = {node["id"]: node for node in graph["nodes"]}
    assert by_id["population-output"]["parents"] == ["population-inputs"]
    identity = 2**54 + 7
    path = tmp_path / "population.zip"
    save_group(path, [(identity, pair.before)], max_bytes=10_000_000)
    loaded = next(load_group(path))
    assert loaded.recipe_json == pair.before.recipe_json


def test_tampered_order_and_failed_construction_rollback():
    from unittest.mock import patch

    recipe = p.realize_recursive_population(request(), seed=5, task_id=7)
    with pytest.raises(ValueError, match="rows differ"):
        replace(recipe, parent_rows=tuple(reversed(recipe.parent_rows)))
    rng = np.random.default_rng(29)
    initial = repr(rng.bit_generator.state)
    with (
        patch.object(
            p.RecursivePopulationRecipe,
            "validate_replay",
            side_effect=ValueError("late custody failure"),
        ),
        pytest.raises(ValueError, match="late custody failure"),
    ):
        p.realize_recursive_population(request(), seed=5, task_id=7, rng=rng)
    assert repr(rng.bit_generator.state) == initial


def test_complete_required_schema_and_numeric_identity_contracts():
    import copy

    recipe = p.realize_recursive_population(request(), seed=5, task_id=7)
    record = asdict(recipe)
    for key in record:
        malformed = copy.deepcopy(record)
        del malformed[key]
        with pytest.raises(ValueError, match="population"):
            p.recipe_from_dict(malformed)
    for key in record["request"]:
        malformed = copy.deepcopy(record)
        del malformed["request"][key]
        with pytest.raises(ValueError, match="population"):
            p.recipe_from_dict(malformed)
    for version in (None, "linear-v1", "unknown-v999"):
        malformed = copy.deepcopy(record)
        malformed["version"] = version
        with pytest.raises(ValueError, match="population"):
            p.recipe_from_dict(malformed)
    for value in (True, 5.0, None, "5"):
        for key in ("seed", "task_id"):
            malformed = copy.deepcopy(record)
            malformed[key] = value
            with pytest.raises(ValueError, match="population"):
                p.recipe_from_dict(malformed)
    malformed = copy.deepcopy(record)
    malformed["request"]["ancestors"]["nodes"][0]["root"].pop("version")
    with pytest.raises(ValueError, match="population"):
        p.recipe_from_dict(malformed)


def test_decode_aggregate_caps_reject_before_snapshot_restore():
    from unittest.mock import patch

    recipe = p.realize_recursive_population(request(), seed=5, task_id=7)
    raw = asdict(recipe)
    with patch.object(
        type(recipe.sample.initial), "restore", side_effect=AssertionError("premature replay")
    ):
        for limits in (
            p.CausalLimits(parameters=50),
            p.CausalLimits(working_bytes=1024**2),
            p.CausalLimits(node_channels=1),
        ):
            with pytest.raises(ValueError, match="limit|workspace"):
                p.recipe_from_dict(raw, limits=limits)


def test_inspection_rejects_semantic_task_and_execution_tamper():
    recipe = p.realize_recursive_population(request(3), seed=5, task_id=7)
    task = p.generate_task(recipe, seed=5, task_id=7)
    with pytest.raises(ValueError, match="population"):
        p.explain_generation(replace(task, task_type="regression"), [0])
    record = json.loads(task.recipe_json)
    record["execution"]["implementation"] = "unrelated-v1"
    with pytest.raises(ValueError, match="population"):
        p.explain_generation(replace(task, recipe_json=json.dumps(record)), [0])


def test_vector_outputs_partial_clamp_preserves_other_coordinates():
    original = request()
    plan = replace(
        original.plan, source="r0-extension-v1", leaf_mean=(1.0, -2.0), leaf_scale=(0.3, 0.7)
    )
    recipe = p.realize_recursive_population(
        replace(original, plan=plan, target_channel=1), seed=5, task_id=7
    )
    before = p.generate_task(recipe, seed=5, task_id=7)
    changed = p.intervene_recursive_population(recipe, forced_value=0.5)
    after = p.generate_task(changed, seed=5, task_id=7)
    torch.testing.assert_close(before.scores[:, 0], after.scores[:, 0], rtol=0, atol=0)
    assert after.targets.numpy().tolist() == [0.5] * 16
    graph = p.explain_generation(after, [15, 0, 15], column_start=1)
    by_id = {n["id"]: n for n in graph["nodes"]}
    assert by_id["population-output"]["parents"] == ["population-before-clamp"]
    np.testing.assert_array_equal(
        np.asarray(by_id["population-before-clamp"]["values"])[:, 0], before.scores[[15, 0, 15], 1]
    )


def test_algorithm4_class_counter_profile_retained():
    original = request(3)
    plan = replace(original.plan, source="mitra-algorithm4-v1", class_assignment="zero-start-cycle")
    recipe = p.realize_recursive_population(replace(original, plan=plan), seed=17, task_id=7)
    result = p.generate_task(recipe, seed=17, task_id=7)
    np.testing.assert_array_equal(result.targets, literal(recipe))
    assert p.recipe_from_dict(asdict(recipe)) == recipe


def test_ancestor_declared_cap_precedes_any_population_restore():
    from unittest.mock import patch

    recipe = p.realize_recursive_population(request(), seed=5, task_id=7)
    raw = asdict(recipe)
    raw["request"]["ancestors"]["limits"]["working_bytes"] = 1024**2
    with (
        patch.object(
            type(recipe.sample.initial), "restore", side_effect=AssertionError("premature replay")
        ),
        pytest.raises(ValueError, match="limit|workspace"),
    ):
        p.recipe_from_dict(raw)
