"""Shared sampled convolution functions keep their complete immutable draw receipt."""

from dataclasses import asdict, replace
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
import torch

from r0.modules import priors as p


def sample():
    request = p.FeatureConv1DRequest(
        3,
        1,
        (
            p.Conv1DLayerPrior(
                (p.Conv1DGeometry(1, 2),),
                activations=(p.CausalActivation(),),
                kernel_laws=("uniform",),
                bias_law="rademacher",
            ),
        ),
    )
    return p.sample_feature_conv1d(
        request, named_rng=np.random.default_rng(51), raw_rng=np.random.default_rng(52)
    )


def test_sampled_shared_function_retains_custody_and_checks_it():
    sampled = sample()
    inputs = torch.arange(12, dtype=torch.float64).reshape(4, 3)
    expected = p.evaluate_feature_conv1d(sampled, inputs)
    assert torch.equal(p.evaluate_frozen_function(sampled, inputs), expected)
    before = asdict(sampled)
    parent = p.FrozenParentFunction((sampled, sampled), "mean")
    assert torch.equal(
        p.evaluate_frozen_function(parent, torch.cat((inputs, inputs), dim=1)), expected
    )
    assert asdict(sampled) == before
    damaged = replace(sampled, choices=((0, 0, 1),))
    with pytest.raises(ValueError, match="replay"):
        p.evaluate_frozen_function(damaged, inputs)


def test_sampled_shared_replay_admission_precedes_snapshot_restore():
    sampled = sample()
    with (
        patch.object(
            type(sampled.initial[0]),
            "restore",
            side_effect=AssertionError("restore before admission"),
        ),
        pytest.raises(ValueError, match="limit|memory|workspace"),
    ):
        p.evaluate_frozen_function(
            sampled,
            torch.zeros((20000, 3), dtype=torch.float64),
            limits=p.CausalLimits(working_bytes=1024),
        )


def test_sampled_graph_generation_retains_raw_source_without_refitting():
    sampled = sample()
    root = p.FunctionalCausalNode(
        (),
        p.FrozenLinearFunction(((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))),
        root=p.JointRootBase("gaussian", 3),
    )
    controls = p.LinearRecipe(
        rows=16, context_rows=8, features=3, signal_features=3, active_features=3
    )
    recipe = p.FunctionalCausalRecipe(
        controls, (root, p.FunctionalCausalNode((0,), sampled)), ((0, 0), (0, 1), (0, 2)), ((1, 0),)
    )
    generated = p.generate_task(recipe, seed=23, task_id=2**54 + 9)
    expected = p.evaluate_feature_conv1d(sampled, generated.inputs)
    assert torch.equal(generated.scores[:, 0], expected[:, 0])
    import json

    assert json.loads(generated.recipe_json)["recipe"]["nodes"][1]["function"] == json.loads(
        json.dumps(asdict(sampled))
    )
    with (
        patch.object(
            type(sampled.initial[0]),
            "restore",
            side_effect=AssertionError("restore before graph admission"),
        ),
        pytest.raises(ValueError, match="limit|memory|workspace"),
    ):
        replace(
            recipe, task=replace(controls, rows=20000), limits=p.CausalLimits(working_bytes=1024)
        )


def test_sampled_convolution_function_and_graph_archive_retain_full_receipt():
    import json

    sampled = sample()
    assert p.frozen_function_from_dict(json.loads(json.dumps(asdict(sampled)))) == sampled
    root = p.FunctionalCausalNode(
        (),
        p.FrozenLinearFunction(((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))),
        root=p.JointRootBase("gaussian", 3),
    )
    controls = p.LinearRecipe(
        rows=16, context_rows=8, features=3, signal_features=3, active_features=3
    )
    recipe = p.FunctionalCausalRecipe(
        controls, (root, p.FunctionalCausalNode((0,), sampled)), ((0, 0), (0, 1), (0, 2)), ((1, 0),)
    )
    assert p.recipe_from_dict(json.loads(json.dumps(asdict(recipe)))) == recipe
    raw = asdict(recipe)
    raw["task"]["rows"] = 20000
    raw["limits"]["working_bytes"] = 8 * 1024**2
    with (
        patch.object(
            type(sampled.initial[0]),
            "restore",
            side_effect=AssertionError("restore before task emission admission"),
        ),
        pytest.raises(ValueError, match="limit|workspace|memory"),
    ):
        p.recipe_from_dict(raw)


def test_sampled_convolution_trace_links_raw_construction_to_realized_kernels():
    sampled = sample()
    root = p.FunctionalCausalNode(
        (),
        p.FrozenLinearFunction(((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))),
        root=p.JointRootBase("gaussian", 3),
    )
    recipe = p.FunctionalCausalRecipe(
        p.LinearRecipe(rows=16, context_rows=8, features=3, signal_features=3, active_features=3),
        (root, p.FunctionalCausalNode((0,), sampled)),
        ((0, 0), (0, 1), (0, 2)),
        ((1, 0),),
    )
    task = p.generate_task(recipe, seed=23, task_id=2**54 + 9)
    graph = p.explain_generation(task, [15, 0, 15])
    nodes = {n["id"]: n for n in graph["nodes"]}
    parameters = "convolution-1-construction-parameters-0"
    assert nodes[parameters]["parents"] == [
        f"convolution-1-construction-draw-{i}" for i in range(5)
    ]
    assert parameters in nodes["convolution-1-layer-0-affine"]["parents"]
    for i, raw in enumerate(sampled.raw):
        np.testing.assert_array_equal(
            nodes[f"convolution-1-construction-draw-{i}"]["values"],
            np.array(raw.values).reshape(1, -1),
        )
    kernel = np.array(sampled.raw[3].values) / np.sqrt(2)
    bias = 0.2 * (2 * np.array(sampled.raw[4].values) - 1)
    np.testing.assert_allclose(
        nodes[parameters]["values"], np.concatenate((kernel, bias)).reshape(1, -1)
    )
    expected = task.inputs.numpy()[[15, 0, 15]]
    expected = expected[:, :-1] * kernel[0] + expected[:, 1:] * kernel[1] + bias[0]
    np.testing.assert_allclose(nodes["convolution-1-layer-0-affine"]["values"], expected)


def test_combined_sampled_convolution_replay_budget_precedes_any_snapshot():
    sampled = sample()
    cap = int(sampled.request.admit()["estimated_working_bytes"] * 1.5)
    limits = p.CausalLimits(working_bytes=cap)
    p.evaluate_frozen_function(sampled, torch.zeros((16, 3), dtype=torch.float64), limits=limits)
    parent = p.FrozenParentFunction((sampled, sampled), "mean")
    with (
        patch.object(
            type(sampled.initial[0]),
            "restore",
            side_effect=AssertionError("restore before aggregate budget"),
        ),
        pytest.raises(ValueError, match="limit|workspace|memory"),
    ):
        p.frozen_function_from_dict(asdict(parent), limits=limits)


@pytest.mark.parametrize("kind", ["native", "vector", "converted"])
def test_sampled_convolution_typed_receipts_traces_and_whole_row_preflight(kind: str):
    import json

    sampled = sample()
    equation = p.FunctionalCausalNode((), sampled, root=p.JointRootBase("gaussian", 3))
    controls = p.LinearRecipe(
        rows=16, context_rows=8, features=1, signal_features=1, active_features=1
    )
    if kind == "native":
        recipe = p.NativeCausalRecipe(controls, (p.NativeCausalNode(equation),), ((0, 0),), (0, 1))
    elif kind == "vector":
        recipe = p.NativeVectorRecipe(
            controls,
            (p.NativeVectorNode(equation, (p.VariableDomain(), p.VariableDomain())),),
            ((0, 0),),
            (0, 1),
        )
    else:
        recipe = p.ConvertedCausalRecipe(
            controls,
            (p.ConvertedCausalNode(equation, ()),),
            (p.ConvertedReadout(0, channel=0),),
            p.ConvertedReadout(0, channel=1),
        )
    raw = json.loads(json.dumps(asdict(recipe)))
    reopened = p.recipe_from_dict(raw)
    assert reopened == recipe
    task = p.generate_task(reopened, seed=7, task_id=2**54 + 3)
    trace = p.explain_generation(task, [15, 0, 15])
    nodes = {n["id"]: n for n in trace["nodes"]}
    assert (
        "convolution-0-construction-parameters-0"
        in nodes["convolution-0-layer-0-affine"]["parents"]
    )
    expected = p.evaluate_feature_conv1d(
        sampled, torch.tensor(nodes["root-0"]["values"], dtype=torch.float64)
    )
    np.testing.assert_allclose(task.targets[[15, 0, 15]].numpy(), expected[:, 1].numpy())
    raw["task"]["rows"] = 20000
    raw["limits"]["working_bytes"] = 8 * 1024**2
    with (
        patch.object(
            type(sampled.initial[0]),
            "restore",
            side_effect=AssertionError("replay before typed task emission budget"),
        ),
        pytest.raises(ValueError, match="limit|workspace|memory"),
    ):
        p.recipe_from_dict(raw)


def test_sampled_convolution_zip_and_soft_intervention_preserve_both_receipts(tmp_path: Path):
    import json

    from r0.modules import generation as g

    original = sample()
    replacement = p.sample_feature_conv1d(
        original.request, named_rng=np.random.default_rng(71), raw_rng=np.random.default_rng(72)
    )
    root = p.FunctionalCausalNode(
        (),
        p.FrozenLinearFunction(((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))),
        root=p.JointRootBase("gaussian", 3),
    )
    recipe = p.FunctionalCausalRecipe(
        p.LinearRecipe(rows=16, context_rows=8, features=3, signal_features=3, active_features=3),
        (root, p.FunctionalCausalNode((0,), original)),
        ((0, 0), (0, 1), (0, 2)),
        ((1, 0),),
    )
    pair = p.generate_causal_pair(
        recipe,
        interventions=(p.SoftIntervention(1, function=replacement),),
        seed=19,
        task_id=2**54 + 17,
    )
    expected = p.evaluate_feature_conv1d(replacement, pair.after.inputs)
    assert torch.equal(pair.after.scores[:, 0], expected[:, 0])
    raw = json.loads(pair.after.recipe_json)
    assert raw["recipe"]["base"] == json.loads(json.dumps(asdict(recipe)))
    path = tmp_path / "sampled-conv.zip"
    g.save_group(path, [(2**54 + 17, pair.after)], max_bytes=5_000_000)
    loaded = next(g.load_group(path))
    assert loaded.recipe_json == pair.after.recipe_json
    restored = p.generate_task(p.recipe_from_dict(raw["recipe"]), seed=19, task_id=2**54 + 17)
    assert torch.equal(restored.targets, pair.after.targets)
    trace = p.explain_generation(loaded, [15, 0, 15])
    nodes = {n["id"]: n for n in trace["nodes"]}
    np.testing.assert_array_equal(
        nodes["convolution-1-construction-draw-3"]["values"],
        np.array(replacement.raw[3].values).reshape(1, -1),
    )
