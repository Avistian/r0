"""Prototype banks preserve joint weights through public graph and typed paths."""

import json
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import pytest
import torch

from r0.modules import priors as p


def root(sampled: bool = False):
    bank = p.FrozenPrototypeRoot(((-2.0, -4.0), (1.0, 2.0), (3.0, 6.0)))
    if sampled:
        return p.PrototypeRootBase(
            p.sample_prototype_root(
                p.PrototypeRootPlan(p.JointRootBase("gaussian", 2), 16, 0.25),
                rng=np.random.default_rng(8),
            )
        )
    return p.PrototypeRootBase(bank)


def equation(base: p.PrototypeRootBase):
    return p.FunctionalCausalNode(
        (),
        p.FrozenLinearFunction(((1.0, 0.0), (0.0, 1.0))),
        base,
        version="functional-causal-node-v2",
    )


def task():
    return p.LinearRecipe(rows=16, context_rows=8, features=1, signal_features=1, active_features=1)


@pytest.mark.parametrize("sampled", [False, True])
def test_literal_shared_weight_prefix_and_raw_replay(sampled: bool):
    base = root(sampled)
    rng = np.random.default_rng(19)
    oracle = np.random.default_rng(19)
    counts = oracle.multinomial(16, [1 / base.frozen.count] * base.frozen.count, size=16)
    expected = (counts / 16) @ np.array(base.frozen.prototypes)
    capture = p.RawDrawLedger()
    result = p.sample_functional_prefix(
        p.FunctionalPrefix((equation(base),), rows=16), rng=rng, capture=capture, owner=("probe",)
    )
    np.testing.assert_array_equal(result.states[0], expected)
    raw = next(r for r in capture.freeze() if r.distribution == "multinomial")
    np.testing.assert_array_equal(raw.array(), counts)
    assert p.prototype_root_base_from_dict(asdict(base)) == base


@pytest.mark.parametrize("kind", ["functional", "native", "vector", "converted", "network"])
@pytest.mark.parametrize("sampled", [False, True])
def test_typed_graph_generation_replay_and_original_rows(kind: str, sampled: bool):
    base = root(sampled)
    node = equation(base)
    if kind == "functional":
        recipe = p.FunctionalCausalRecipe(task(), (node,), ((0, 0),), ((0, 1),))
    elif kind == "native":
        # Native scalar continuous nodes retain a vector-valued state.
        recipe = p.NativeCausalRecipe(
            task(), (p.NativeCausalNode(node, p.VariableDomain()),), ((0, 0),), (0, 1)
        )
    elif kind == "vector":
        recipe = p.NativeVectorRecipe(
            task(), (p.NativeVectorNode(node, (p.VariableDomain(),) * 2),), ((0, 0),), (0, 1)
        )
    elif kind == "converted":
        converter = p.CategoricalConverter(
            "side", 0, 1, ("left", "right"), "neighbor_id", ((-1.0,), (1.0,))
        )
        recipe = p.ConvertedCausalRecipe(
            task(),
            (p.ConvertedCausalNode(node, (converter,)),),
            (p.ConvertedReadout(0, capture="side"),),
            p.ConvertedReadout(0, channel=1),
        )
    else:
        recipe = p.NetworkRecipe(
            replace(task(), features=2, signal_features=2, active_features=2),
            p.FrozenNetworkFunction(
                2, (p.NetworkBlock((0,), (p.NetworkAffine(((2.0, 3.0),), (1.0,)),)),)
            ),
            base,
        )
    restored = p.recipe_from_dict(asdict(recipe))
    assert restored == recipe
    emitted = p.generate_task(recipe, seed=29, task_id=2**54 + 3)
    replay = p.generate_task(restored, seed=29, task_id=2**54 + 3)
    assert torch.equal(emitted.targets, replay.targets)
    assert json.loads(emitted.recipe_json)["recipe"] == json.loads(json.dumps(asdict(recipe)))
    trace = p.explain_generation(emitted, [15, 0, 15])
    assert trace["rows"] == [15, 0, 15]
    target_node = next(n for n in trace["nodes"] if n["id"] in ("targets", "target"))
    np.testing.assert_allclose(
        np.array(target_node["values"]).reshape(-1),
        emitted.targets[[15, 0, 15]].numpy().reshape(-1),
    )


def test_batch_reference_is_not_an_independent_graph_root():
    batch = p.sample_prototype_batch_reference(
        p.PrototypeRootPlan(p.JointRootBase("gaussian", 2), 16, 0.25),
        prototype_rng=np.random.default_rng(1),
        weights_rng=np.random.default_rng(2),
    )
    with pytest.raises(ValueError, match="independent"):
        p.PrototypeRootBase(batch)  # type: ignore[arg-type]


def test_mixed_native_vector_variables_share_prototype_weight_source():
    node = p.NativeVectorNode(
        equation(root()),
        (
            p.VariableDomain(),
            p.VariableDomain("binary", ("no", "yes"), "threshold", (0.0,)),
        ),
    )
    recipe = p.NativeVectorRecipe(
        replace(task(), task_type="binary", classes=2), (node,), ((0, 0),), (0, 1)
    )
    emitted = p.generate_task(recipe, seed=4, task_id=7)
    np.testing.assert_array_equal(emitted.targets, emitted.inputs[:, 0] >= 0)
    assert p.recipe_from_dict(asdict(recipe)) == recipe
    assert p.explain_generation(emitted, [15, 0, 15])["rows"] == [15, 0, 15]


def test_frozen_bank_conditional_variance_allows_separate_target_calibration():
    source = p.SourceRootBase(p.SourceRootLaw((p.SourceRootCoordinate("zipf", exponent=1.8),)))
    sampled = p.sample_prototype_root(
        p.PrototypeRootPlan(source, 32, 0.25), rng=np.random.default_rng(8)
    )
    assert sampled.construction_moment_order < 1
    assert sampled.root.moment_order == float("inf")
    node = p.FunctionalCausalNode(
        (),
        p.FrozenLinearFunction(((1.0,),)),
        p.PrototypeRootBase(sampled),
        version="functional-causal-node-v2",
    )
    child = p.FunctionalCausalNode((0,), p.FrozenLinearFunction(((2.0,),)))
    recipe = p.CalibratedFunctionalCausalRecipe(task(), (node, child), ((0, 0),), ((1, 0),))
    emitted = p.generate_task(recipe, seed=13, task_id=2)
    assert torch.isfinite(emitted.targets).all()
    assert p.recipe_from_dict(asdict(recipe)) == recipe


def test_prototype_graph_budget_and_bank_tamper_fail_closed():
    base = root(True)
    node = equation(base)
    with pytest.raises(ValueError, match="limit|memory"):
        p.FunctionalPrefix((node,), rows=1024, limits=p.CausalLimits(working_bytes=10000))
    payload = asdict(base)
    payload["bank"]["raw"] = ()
    with pytest.raises(ValueError, match="raw|prototype"):
        p.prototype_root_base_from_dict(payload)


def test_prototype_plan_preserves_sampled_source_selection_and_independent_fit():
    profile = p.sample_source_root_profile(
        p.SourceRootProfile(2, selector="forced-r0", families=("gaussian", "zipf_clipped")),
        model_rng=np.random.default_rng(1),
        selection_rng=np.random.default_rng(2),
    )
    source = p.sample_source_root_base(
        profile,
        modes=("standardize", "center"),
        calibration_rows=32,
        calibration_rng=np.random.default_rng(3),
    )
    sampled = p.sample_prototype_root(
        p.PrototypeRootPlan(source, 16, 0.25), rng=np.random.default_rng(4)
    )
    base = p.PrototypeRootBase(sampled)
    assert p.prototype_root_base_from_dict(asdict(base)) == base
    recipe = p.FunctionalCausalRecipe(task(), (equation(base),), ((0, 0),), ((0, 1),))
    assert p.recipe_from_dict(asdict(recipe)) == recipe
    emitted = p.generate_task(recipe, seed=11, task_id=5)
    assert p.explain_generation(emitted, [15, 0, 15])["signal_max_error"] == 0


def test_prototype_network_matching_keeps_independent_population():
    source = p.NetworkRecipe(
        replace(task(), features=2, signal_features=2, active_features=2),
        p.FrozenNetworkFunction(
            2, (p.NetworkBlock((0,), (p.NetworkAffine(((2.0, 3.0),), (1.0,)),)),)
        ),
        root(),
    )
    matching = p.fit_network_variance_match(
        source, before_inputs=(0, 1), after_inputs=(1,), rows=64, rng=np.random.default_rng(8)
    )
    view = p.MatchedNetworkInputView(
        source, replace(source.task, active_features=1), (1,), matching=matching
    )
    assert p.recipe_from_dict(asdict(view)) == view
    emitted = p.generate_task(view, seed=8, task_id=4)
    assert p.explain_generation(emitted, [15, 0, 15])["signal_max_error"] == 0


def test_prototype_pair_persistence_preserves_bank_and_original_rows(tmp_path: Path):
    from r0.modules.generation import load_causal_pair, save_causal_pair

    recipe = p.FunctionalCausalRecipe(
        task(),
        (equation(root(True)), p.FunctionalCausalNode((0,), p.FrozenLinearFunction(((2.0, 3.0),)))),
        ((0, 0),),
        ((1, 0),),
    )
    pair = p.generate_causal_pair(
        recipe, interventions=(p.HardIntervention(1, (0,), (4.0,)),), seed=8, task_id=4
    )
    save_causal_pair(tmp_path / "pair", pair, max_bytes=4_000_000)
    restored = load_causal_pair(tmp_path / "pair")
    assert restored.before.recipe_json == pair.before.recipe_json
    assert restored.after.episode() == pair.after.episode()
    assert p.explain_generation(restored.after, [15, 0, 15])["rows"] == [15, 0, 15]


def test_prototype_workspace_is_aggregated_across_roots_before_evaluation():
    limits = p.CausalLimits(parameters=1000)
    node = equation(root())
    assert p.FunctionalPrefix((node,), rows=32, limits=limits).rows == 32
    with pytest.raises(ValueError, match="aggregate|parameter|limit"):
        p.FunctionalPrefix((node, node), rows=32, limits=limits)
