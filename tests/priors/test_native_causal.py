"""Native variables retain their domains through equations, tasks and archives."""

import json
from dataclasses import asdict, replace
from pathlib import Path
from typing import Literal, cast

import pytest
import torch

from r0.modules.generation import load_group, save_group
from r0.modules.priors import (
    FrozenLinearFunction,
    FunctionalCausalNode,
    JointRootBase,
    LinearRecipe,
    generate_task,
    recipe_from_dict,
)
from r0.modules.priors.lib.native_causal import NativeCausalNode, NativeCausalRecipe, VariableDomain
from r0.modules.priors.lib.native_evaluation import evaluate_native_graph


def mixed_recipe(emission: Literal["softmax", "argmax"] = "softmax") -> NativeCausalRecipe:
    categorical = VariableDomain("categorical", ("red", "green", "blue"), emission)
    binary = VariableDomain("binary", ("no", "yes"), "threshold", (0.0,))
    return NativeCausalRecipe(
        LinearRecipe(
            rows=32,
            features=3,
            signal_features=3,
            active_features=3,
            context_rows=20,
            task_type="multiclass",
            classes=3,
        ),
        (
            NativeCausalNode(
                FunctionalCausalNode(
                    (), FrozenLinearFunction(((1.0,),)), JointRootBase("gaussian", 1)
                )
            ),
            NativeCausalNode(FunctionalCausalNode((0,), FrozenLinearFunction(((1.0,),))), binary),
            NativeCausalNode(FunctionalCausalNode((1,), FrozenLinearFunction(((10.0, 20.0),)))),
            NativeCausalNode(
                FunctionalCausalNode((0,), FrozenLinearFunction(((-1.0,), (0.0,), (1.0,)))),
                categorical,
            ),
            NativeCausalNode(
                FunctionalCausalNode(
                    (3,),
                    FrozenLinearFunction(((1.0, 0.0, -1.0), (0.0, 1.0, 0.0), (-1.0, 0.0, 1.0))),
                ),
                categorical,
            ),
        ),
        ((1, 0), (2, 0), (3, 0)),
        (4, 0),
    )


def test_native_state_is_integer_and_children_receive_one_hot() -> None:
    recipe = mixed_recipe()
    result = evaluate_native_graph(recipe, seed=19, task_id=2)
    assert result.values[1].dtype == torch.int64
    assert result.values[3].dtype == torch.int64
    assert torch.equal(result.values[2][:, 0], 10.0 + 10.0 * result.values[1][:, 0])
    assert torch.equal(result.values[1][:, 0], (result.values[0][:, 0] >= 0).long())
    task = generate_task(recipe, seed=19, task_id=2)
    assert torch.equal(task.targets, result.values[4][:, 0])
    assert task.target_origin == "native-variable-v1"
    assert json.loads(task.mechanism_json)["native_emissions"]["4"]["uniforms"] == (
        result.uniforms[4].numpy().tolist()
    )


def test_mixed_episode_and_archive_keep_domains(tmp_path: Path) -> None:
    task = generate_task(mixed_recipe(), seed=19, task_id=2)
    episode = task.episode()
    assert tuple(f.kind for f in episode.features) == ("binary", "continuous", "categorical")
    assert all(type(row.values[0]) is int for row in (*episode.context, *episode.queries))
    assert all(row.values[2] in ("red", "green", "blue") for row in episode.context)
    assert episode.classes == ("red", "green", "blue")
    path = tmp_path / "native.zip"
    save_group(path, [(2, task)], max_bytes=1_000_000)
    loaded = next(load_group(path))
    assert loaded.episode() == episode
    assert loaded.feature_schema == task.feature_schema
    assert loaded.target_origin == task.target_origin
    assert loaded.mechanism_json == task.mechanism_json


def test_roundtrip_and_resplit_preserve_native_values_and_source() -> None:
    recipe = mixed_recipe("argmax")
    decoded = recipe_from_dict(json.loads(json.dumps(asdict(recipe))))
    assert decoded == recipe
    original = generate_task(recipe, seed=6, task_id=3)
    replay = generate_task(decoded, seed=6, task_id=3)
    assert replay.recipe_json == original.recipe_json
    assert torch.equal(replay.targets, original.targets)
    resplit = generate_task(
        replace(recipe, task=replace(recipe.task, split=2, context_rows=10)), seed=6, task_id=3
    )
    assert resplit.source_task_id == original.source_task_id
    assert resplit.episode_id != original.episode_id
    assert torch.equal(resplit.inputs, original.inputs)
    assert torch.equal(resplit.targets, original.targets)


def test_domains_and_native_target_compatibility_fail_closed() -> None:
    with pytest.raises(ValueError, match="vocabulary"):
        VariableDomain("categorical", ("same", "same"), "argmax")
    with pytest.raises(ValueError, match="threshold"):
        VariableDomain("binary", ("no", "yes"), "threshold", (1.0, 2.0))
    recipe = mixed_recipe()
    with pytest.raises(ValueError, match="target"):
        replace(recipe, task=replace(recipe.task, task_type="binary", classes=2))
    with pytest.raises(ValueError, match="adapter"):
        replace(recipe, task=replace(recipe.task, separation=2.0))


def test_softmax_uses_retained_uniforms_and_fixed_declared_vocabulary() -> None:
    import numpy as np

    recipe = mixed_recipe()
    result = evaluate_native_graph(recipe, seed=7, task_id=8)
    parent = result.values[3][:, 0].numpy()
    logits = np.array([[1.0, 0.0, -1.0], [0.0, 1.0, 0.0], [-1.0, 0.0, 1.0]])[parent]
    probabilities = np.exp(logits) / np.exp(logits).sum(axis=1, keepdims=True)
    uniform = result.uniforms[4].numpy()
    expected = np.array(
        [
            int(cast(np.intp, np.searchsorted(np.cumsum(row), u, side="right")))
            for row, u in zip(probabilities, uniform, strict=True)
        ],
        dtype=np.int64,
    )
    assert np.array_equal(result.values[4][:, 0].numpy(), expected)
    np.testing.assert_allclose(result.probabilities[4].numpy(), probabilities, rtol=1e-14)


def test_argmax_tie_and_threshold_boundary_are_declared() -> None:
    recipe = mixed_recipe("argmax")
    nodes = list(recipe.nodes)
    nodes[0] = replace(
        nodes[0], equation=replace(nodes[0].equation, function=FrozenLinearFunction(((0.0,),)))
    )
    result = evaluate_native_graph(replace(recipe, nodes=tuple(nodes)), seed=7, task_id=8)
    assert torch.all(result.values[1] == 1)  # threshold equality goes to the upper interval
    assert torch.all(result.values[3] == 0)  # first-index argmax tie
    task = generate_task(replace(recipe, nodes=tuple(nodes)), seed=7, task_id=8)
    assert task.classes == ("red", "green", "blue")  # never inferred from emitted rows
    assert task.feature_schema[2].vocabulary == task.classes


def test_inspection_preserves_order_repeats_and_rejects_changed_uniform() -> None:
    from r0.modules.priors import explain_generation

    task = generate_task(mixed_recipe(), seed=3, task_id=1)
    graph = explain_generation(task, [9, 1, 9])
    target = next(node for node in graph["nodes"] if node["id"] == "targets")
    assert target["values"] == [[float(task.targets[i])] for i in (9, 1, 9)]
    assert graph["target_origin"] == "native-variable-v1"
    mechanism = json.loads(task.mechanism_json)
    mechanism["native_emissions"]["4"]["uniforms"][0] = 0.5
    with pytest.raises(ValueError, match="variates"):
        explain_generation(replace(task, mechanism_json=json.dumps(mechanism)), [1])


def test_native_admission_charges_emission_storage_and_encoded_width() -> None:
    from r0.modules.priors import CausalLimits

    recipe = mixed_recipe()
    estimate = recipe.admit()["estimated_working_bytes"]
    with pytest.raises(ValueError, match="working memory"):
        replace(recipe, limits=replace(CausalLimits(), working_bytes=estimate - 1))
    with pytest.raises(ValueError, match="channel"):
        replace(recipe, limits=replace(CausalLimits(), node_channels=2))
    nodes = list(recipe.nodes)
    nodes[2] = replace(
        nodes[2], equation=replace(nodes[2].equation, function=FrozenLinearFunction(((1.0,),)))
    )
    with pytest.raises(ValueError, match="encoded parent"):
        replace(recipe, nodes=tuple(nodes))


def test_native_regression_and_binary_targets_do_not_use_label_adapter() -> None:
    recipe = mixed_recipe()
    for target, kind, classes in (((2, 0), "regression", 0), ((1, 0), "binary", 2)):
        controls = replace(
            recipe.task,
            features=1,
            signal_features=1,
            active_features=1,
            task_type=kind,
            classes=classes,
        )
        variant = replace(recipe, task=controls, observed=((0, 0),), target=target)
        result = evaluate_native_graph(variant, seed=7, task_id=1)
        task = generate_task(variant, seed=7, task_id=1)
        assert torch.equal(task.targets, result.values[target[0]][:, 0])
        assert task.episode().task_type == kind
        assert task.target_origin == "native-variable-v1"


def test_typed_transport_rejects_fractional_or_out_of_vocabulary_codes() -> None:
    task = generate_task(mixed_recipe(), seed=3, task_id=1)
    bad_inputs = task.inputs.clone()
    bad_inputs[0, 2] = 0.5
    with pytest.raises(ValueError, match="feature codes"):
        replace(task, inputs=bad_inputs)
    with pytest.raises(ValueError, match="int64"):
        replace(task, targets=task.targets.to(torch.float64))
    with pytest.raises(ValueError, match="schema"):
        replace(task, feature_schema=())


def test_native_vector_continuous_target_selects_its_score_channel() -> None:
    recipe = mixed_recipe()
    nodes = list(recipe.nodes)
    nodes[2] = replace(
        nodes[2],
        equation=replace(
            nodes[2].equation, function=FrozenLinearFunction(((10.0, 20.0), (30.0, 40.0)))
        ),
    )
    task_controls = replace(
        recipe.task,
        task_type="regression",
        classes=0,
        features=1,
        signal_features=1,
        active_features=1,
    )
    variant = replace(
        recipe, nodes=tuple(nodes), task=task_controls, observed=((0, 0),), target=(2, 1)
    )
    task = generate_task(variant, seed=4, task_id=5)
    assert task.scores.shape == (32, 1)
    assert torch.equal(task.scores[:, 0], task.targets)
    assert set(task.targets.numpy().tolist()) <= {30.0, 40.0}
