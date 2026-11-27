"""Observation source custody reaches public task archives, traces and interventions."""

from __future__ import annotations

import json
from dataclasses import asdict, replace
from pathlib import Path
from typing import Literal

import numpy as np
import pytest
import torch

from r0.modules import generation as g
from r0.modules import priors as p


def authored_source() -> p.NetworkRecipe:
    task = p.LinearRecipe(
        rows=16,
        context_rows=7,
        features=2,
        signal_features=2,
        active_features=2,
        task_type="multiclass",
        classes=4,
    )
    first = p.NetworkBlock(
        (0,), (p.NetworkAffine(((1.0, 0.0), (0.0, 1.0), (1.0, 1.0), (1.0, -1.0)), (0.0,) * 4),)
    )
    matrix = tuple(tuple(float(i == j) * 2 for j in range(4)) for i in range(4))
    blocks = (
        first,
        *(p.NetworkBlock((i,), (p.NetworkAffine(matrix, (float(i),) * 4),)) for i in (1, 2)),
    )
    return p.NetworkRecipe(task, p.FrozenNetworkFunction(2, blocks))


def source_variant(kind: str) -> p.NetworkRecipe | p.NetworkInnovationRecipe:
    source = authored_source()
    if kind == "sampled-network":
        sampled = p.sample_network(
            p.NetworkSamplingRequest(2, (4, 4, 4), families=("dense",)),
            named_rng=np.random.default_rng(51),
            raw_rng=np.random.default_rng(52),
        )
        return replace(source, network=sampled)
    if kind == "hyperprior":
        request = p.NetworkHyperpriorRequest(
            p.NetworkSamplingRequest(2, (4,), families=("dense",)),
            depth=p.NetworkScalarPrior("fixed", 2.0, 2.0),
            width=p.NetworkScalarPrior("fixed", 4.0, 4.0),
            weight_scale=p.NetworkScalarPrior("fixed", 0.4, 0.4),
            max_hidden_blocks=2,
            max_hidden_width=4,
        )
        sampled = p.sample_hyperprior_network(
            request,
            hyper_rng=np.random.default_rng(50),
            named_rng=np.random.default_rng(51),
            raw_rng=np.random.default_rng(52),
        )
        return replace(source, network=sampled)
    if kind == "statistics":
        return p.sample_network_innovations(
            source,
            p.NetworkInnovationRequest(
                p.NetworkScalarPrior("uniform", 0.1, 0.3),
                importance=p.NetworkScalarPrior("uniform", 0.5, 1.5),
            ),
            statistics_rng=np.random.default_rng(53),
        )
    if kind in ("sampled-root", "prototype-sampled-root"):
        profile = p.sample_source_root_profile(
            p.SourceRootProfile(2, selector="forced-r0", families=("gaussian", "unit_uniform")),
            model_rng=np.random.default_rng(54),
            selection_rng=np.random.default_rng(55),
        )
        root = p.sample_source_root_base(
            profile,
            modes=("standardize", "center"),
            calibration_rows=16,
            calibration_rng=np.random.default_rng(56),
        )
        source = replace(source, root=root)
    if kind in ("prototype", "prototype-sampled-root"):
        base = source.root or p.JointRootBase("gaussian", 2)
        assert isinstance(base, (p.JointRootBase, p.SourceRootBase))
        bank = p.sample_prototype_root(
            p.PrototypeRootPlan(base, 16, 0.5),
            rng=np.random.default_rng(57),
        )
        return replace(source, root=p.PrototypeRootBase(bank))
    return source


def envelope(
    source: p.NetworkRecipe | p.NetworkInnovationRecipe,
    *,
    profile: str = "p1-778f0d0-v1",
    target: str = "final",
    selection: str = "block",
    task_type: Literal["regression", "binary", "multiclass"] = "regression",
) -> p.NetworkObservationRecipe:
    outputs = 3 if task_type == "multiclass" else 1
    receipt = p.sample_network_observations(
        p.NetworkObservationRequest(
            source,
            2,
            outputs=outputs,
            profile=profile,
            target=target,
            selection=selection,
            feature_order="permuted",
        ),
        rng=np.random.default_rng(2),
    )
    task = p.LinearRecipe(
        rows=16,
        context_rows=7,
        features=2,
        signal_features=2,
        active_features=2,
        task_type=task_type,
        classes=3 if task_type == "multiclass" else 2 if task_type == "binary" else 0,
    )
    return p.NetworkObservationRecipe(task, receipt)


@pytest.mark.parametrize("profile", ["p1-paper-block-r0-v1", "p1-778f0d0-v1"])
@pytest.mark.parametrize("selection", ["block", "unrestricted"])
@pytest.mark.parametrize("target", ["internal", "final"])
@pytest.mark.parametrize("task_type", ["regression", "binary", "multiclass"])
def test_literal_channel_selection_task_archive_and_trace(
    profile: str,
    selection: str,
    target: str,
    task_type: Literal["regression", "binary", "multiclass"],
    tmp_path: Path,
) -> None:
    source = authored_source()
    recipe = envelope(
        source, profile=profile, selection=selection, target=target, task_type=task_type
    )
    tid = 2**54 + 29
    baseline = p.generate_task(source, seed=73, task_id=tid)
    x = baseline.inputs.numpy()
    first = np.column_stack((x[:, 0], x[:, 1], x[:, 0] + x[:, 1], x[:, 0] - x[:, 1]))
    second = 2 * first + 1
    values = np.concatenate((first, second, 2 * second + 2), axis=1)
    if profile == "p1-778f0d0-v1":
        values = values[:, 4:]
    result = p.generate_task(recipe, seed=73, task_id=tid)
    np.testing.assert_allclose(
        result.inputs.numpy(), values[:, recipe.observations.features], rtol=0, atol=1e-12
    )
    np.testing.assert_allclose(
        result.scores.numpy(), values[:, recipe.observations.targets], rtol=0, atol=1e-12
    )
    raw = json.loads(result.recipe_json)["recipe"]
    assert raw == json.loads(json.dumps(asdict(recipe)))
    assert p.recipe_from_dict(raw) == recipe
    path = tmp_path / "selected.zip"
    g.save_group(path, [(tid, result)], max_bytes=8 * 1024**2)
    loaded = next(g.load_group(path))
    replay = p.generate_task(
        p.recipe_from_dict(json.loads(loaded.recipe_json)["recipe"]), seed=73, task_id=tid
    )
    assert torch.equal(replay.targets, result.targets)
    assert loaded.recipe_json == result.recipe_json
    trace = p.explain_generation(loaded, [15, 0, 15])
    assert trace["signal_max_error"] == 0
    assert trace["observation_selection"]["features"] == list(recipe.observations.features)
    for column, (node, channel) in enumerate(recipe.runtime.observed):
        readout = next(n for n in trace["nodes"] if n["id"] == f"readout-{node}")
        np.testing.assert_allclose(
            np.array(readout["values"])[:, channel], result.inputs[[15, 0, 15], column].numpy()
        )


@pytest.mark.parametrize(
    "kind",
    [
        "sampled-network",
        "hyperprior",
        "statistics",
        "sampled-root",
        "prototype",
        "prototype-sampled-root",
    ],
)
def test_nested_source_raw_custody_roundtrip_and_prefix(kind: str) -> None:
    recipe = envelope(source_variant(kind))
    original = json.dumps(asdict(recipe.observations), sort_keys=True)
    reopened = p.recipe_from_dict(json.loads(json.dumps(asdict(recipe))))
    assert reopened == recipe
    before = p.generate_task(recipe, seed=17, task_id=2**54 + 47)
    after = p.generate_task(
        replace(recipe, task=replace(recipe.task, rows=24)), seed=17, task_id=2**54 + 47
    )
    exact = p.generate_task(reopened, seed=17, task_id=2**54 + 47)
    assert torch.equal(before.inputs, exact.inputs)
    assert torch.equal(before.scores, exact.scores)
    if kind != "sampled-root":
        assert torch.equal(before.inputs, after.inputs[:16])
        assert torch.equal(before.scores, after.scores[:16])
    else:
        # Existing root v1 consumes whole coordinate arrays sequentially, so shape changes
        # shift later-coordinate streams. Its frozen law/calibration remains unchanged.
        source = recipe.observations.request.source
        assert isinstance(source, p.NetworkRecipe)
        expanded = p.generate_task(
            replace(source, task=replace(source.task, rows=24)), seed=17, task_id=2**54 + 47
        )
        x = expanded.inputs.numpy()
        first = np.column_stack((x[:, 0], x[:, 1], x[:, 0] + x[:, 1], x[:, 0] - x[:, 1]))
        second = 2 * first + 1
        literal = np.concatenate((second, 2 * second + 2), axis=1)
        np.testing.assert_allclose(
            after.inputs.numpy(), literal[:, recipe.observations.features], rtol=0, atol=1e-12
        )
        np.testing.assert_allclose(
            after.scores.numpy(), literal[:, recipe.observations.targets], rtol=0, atol=1e-12
        )
    assert json.dumps(asdict(recipe.observations), sort_keys=True) == original
    trace = p.explain_generation(before, [15, 0, 15])
    assert trace["signal_max_error"] == 0


@pytest.mark.parametrize("location", ["state", "readout", "state_and_readout"])
def test_overlap_intervention_stage_semantics_and_original_custody(location: str) -> None:
    recipe = envelope(authored_source())
    selected = recipe.observations.targets[0]
    ref = recipe.observations.catalog[selected]
    change = p.HardIntervention(
        ref.node, channels=(ref.channel,), values=(7.25,), location=location
    )
    pair = p.generate_causal_pair(recipe, interventions=(change,), seed=31, task_id=2**54 + 51)
    before = json.loads(pair.before.recipe_json)["recipe"]
    after = json.loads(pair.after.recipe_json)["recipe"]
    assert after["base"] == before == json.loads(json.dumps(asdict(recipe)))
    feature = recipe.observations.features.index(selected)
    np.testing.assert_array_equal(pair.after.inputs[:, feature].numpy(), [7.25] * 16)
    if location == "readout":
        assert torch.equal(pair.before.targets, pair.after.targets)
    else:
        np.testing.assert_array_equal(pair.after.targets.numpy(), [7.25] * 16)
    assert p.explain_generation(pair.after, [15, 0, 15])["rows"] == [15, 0, 15]
    reopened = p.recipe_from_dict(after)
    assert torch.equal(
        p.generate_task(reopened, seed=31, task_id=2**54 + 51).targets, pair.after.targets
    )


def test_decoder_aggregate_rows_reject_before_any_source_snapshot(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = source_variant("sampled-root")
    assert isinstance(source, p.NetworkRecipe)
    recipe = envelope(replace(source, limits=p.CausalLimits(working_bytes=32 * 1024**2)))
    raw = asdict(recipe)
    raw["task"]["rows"] = 20000

    def forbidden(*args: object, **kwargs: object) -> None:
        raise AssertionError("source snapshot restored before complete envelope admission")

    monkeypatch.setattr(type(recipe.observations.initial), "restore", forbidden)
    with pytest.raises(ValueError, match="limit|memory"):
        p.recipe_from_dict(raw)


@pytest.mark.parametrize("part", ["features", "catalog", "raw", "source", "version", "extra"])
def test_source_selection_or_envelope_tamper_rejects(part: str) -> None:
    recipe = envelope(authored_source())
    raw = json.loads(json.dumps(asdict(recipe)))
    receipt = raw["observations"]
    if part == "features":
        receipt["features"][0] = True
    elif part == "catalog":
        receipt["catalog"][0]["node"] = 0
    elif part == "raw":
        receipt["raw"][0]["values"][0] = 0
    elif part == "source":
        receipt["request"]["source"]["network"]["blocks"][0]["experts"][0]["matrix"][0][0] = 17.0
        # Authored source edits are allowed; the old generated task/card must not survive.
        changed = p.recipe_from_dict(raw)
        original = p.generate_task(recipe, seed=17, task_id=9)
        with pytest.raises(ValueError, match="metadata|signal"):
            p.explain_generation(
                replace(
                    original,
                    recipe_json=json.dumps(
                        {**json.loads(original.recipe_json), "recipe": asdict(changed)}
                    ),
                ),
                [0],
            )
        return
    elif part == "version":
        raw["version"] = "network-observation-recipe-v2"
    else:
        raw["nodes"] = []
    with pytest.raises(ValueError, match="observation|network|recipe|channel"):
        p.recipe_from_dict(raw)


def test_caller_caps_and_oversized_metadata_precede_snapshot_restore(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    recipe = envelope(source_variant("sampled-network"))
    raw = asdict(recipe)

    def forbidden(*args: object, **kwargs: object) -> None:
        raise AssertionError("metadata/caller admission must precede source replay")

    monkeypatch.setattr(type(recipe.observations.initial), "restore", forbidden)
    with pytest.raises(ValueError, match="limits"):
        p.recipe_from_dict(
            raw, limits=p.CausalLimits(working_bytes=recipe.limits.working_bytes - 1)
        )
    raw["observations"]["request"]["source"]["network"]["raw"] = [0.0] * 500
    with pytest.raises(ValueError, match="metadata|limit"):
        p.recipe_from_dict(raw, limits=p.CausalLimits(parameters=300))


def test_selected_equation_and_root_replacement_retain_selection() -> None:
    recipe = envelope(authored_source())
    ref = recipe.observations.catalog[recipe.observations.targets[0]]
    node = recipe.runtime.nodes[ref.node]
    width = node.function.input_width
    function = p.FrozenLinearFunction(
        tuple((0.0,) * width for _ in range(node.function.output_width)),
    )
    pair = p.generate_causal_pair(
        recipe, interventions=(p.SoftIntervention(ref.node, function=function),), seed=31, task_id=4
    )
    np.testing.assert_array_equal(pair.after.scores.numpy(), np.zeros((16, 1)))
    trace = p.explain_generation(pair.after, [15, 0, 15])
    assert trace["observation_selection"]["target_aliases"]
    assert json.loads(pair.after.mechanism_json)["observation_selection"]["features"] == list(
        recipe.observations.features
    )
    bad_card = json.loads(pair.after.mechanism_json)
    bad_card["observation_selection"]["target_aliases"] = []
    with pytest.raises(ValueError, match="observation metadata"):
        p.explain_generation(replace(pair.after, mechanism_json=json.dumps(bad_card)), [0])
    root = p.JointRootBase("uniform", 2)
    changed = p.generate_causal_pair(
        recipe, interventions=(p.StageIntervention(0, root=root),), seed=31, task_id=4
    )
    decoded = p.recipe_from_dict(json.loads(changed.after.recipe_json)["recipe"])
    assert torch.equal(p.generate_task(decoded, seed=31, task_id=4).scores, changed.after.scores)
    assert json.loads(changed.after.recipe_json)["recipe"]["base"] == json.loads(
        json.dumps(asdict(recipe))
    )
