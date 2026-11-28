"""Public literal architecture arithmetic, frozen sampling and archive custody."""

import json
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import pytest
import torch

from r0.modules import priors as p


def test_irregular_gated_projected_skip_literal_arithmetic():
    # Input state 0; two branches see the same concatenated inputs.
    a = p.NetworkAffine(((2.0, -1.0),), (0.5,), activation=p.CausalActivation("tanh"))
    b = p.NetworkAffine(((-1.0, 3.0),), (-0.2,), mask=((True, False),))
    gate = p.NetworkAffine(((1.0, 0.0), (0.0, -2.0)), (0.1, -0.3))
    skip = p.NetworkProjection(0, p.NetworkAffine(((0.4, 0.8),), (0.0,)))
    function = p.FrozenNetworkFunction(2, (p.NetworkBlock((0,), (a, b), gate, (skip,)),))
    x = np.array([[-1.0, 2.0], [0.3, -0.7], [3.0, 1.0]])
    logits = x @ np.array(gate.matrix).T + gate.bias
    weights = np.exp(logits - logits.max(1, keepdims=True))
    weights /= weights.sum(1, keepdims=True)
    expected = (
        weights[:, 0] * np.tanh(x @ np.array([2.0, -1.0]) + 0.5)
        + weights[:, 1] * (-x[:, 0] - 0.2)
        + x @ np.array([0.4, 0.8])
    )
    actual = p.evaluate_frozen_function(function, torch.tensor(x))
    np.testing.assert_allclose(actual[:, 0], expected, rtol=1e-13, atol=1e-13)
    assert p.frozen_function_from_dict(asdict(function)) == function


@pytest.mark.parametrize(
    "family",
    ["dense", "individual-sparse", "block-sparse", "skip", "residual", "irregular", "gated"],
)
def test_sampled_network_standalone_replay_and_prefix(family: str):
    request = p.NetworkSamplingRequest(3, (5, 4, 3), families=(family,), dropout=0.4, experts=3)
    graph = p.sample_network(
        request, named_rng=np.random.default_rng(7), raw_rng=np.random.default_rng(8)
    )
    graph.validate_replay()
    task = p.LinearRecipe(
        rows=16,
        context_rows=7,
        features=3,
        signal_features=3,
        active_features=3,
        task_type="multiclass",
        classes=3,
    )
    recipe = p.NetworkRecipe(task, graph)
    result = p.generate_task(recipe, seed=12, task_id=9007199254740995)
    decoded = p.recipe_from_dict(json.loads(result.recipe_json)["recipe"])
    assert torch.equal(
        p.generate_task(decoded, seed=12, task_id=9007199254740995).targets, result.targets
    )
    larger = p.generate_task(
        replace(recipe, task=replace(task, rows=32, context_rows=5)),
        seed=12,
        task_id=9007199254740995,
    )
    torch.testing.assert_close(result.inputs, larger.inputs[:16], rtol=0, atol=0)
    torch.testing.assert_close(result.scores, larger.scores[:16], rtol=0, atol=0)
    inspection = p.explain_generation(result, [15, 0, 15])
    assert any("expert" in node["id"] for node in inspection["nodes"])


def test_cycle_and_ordered_geometry_reject():
    affine = p.NetworkAffine(((1.0,),), (0.0,))
    with pytest.raises(ValueError, match="earlier"):
        p.FrozenNetworkFunction(1, (p.NetworkBlock((1,), (affine,)),))
    with pytest.raises(ValueError, match="width"):
        p.FrozenNetworkFunction(2, (p.NetworkBlock((0,), (affine,)),))


@pytest.mark.parametrize("exponent", [0.5, 1.0])
@pytest.mark.parametrize("profile", ["p1-individual", "p1-block"])
def test_p1_named_mask_scaling_and_first_layer(exponent: float, profile: str):
    family = "individual-sparse" if profile == "p1-individual" else "block-sparse"
    req = p.NetworkSamplingRequest(
        5,
        (7, 6, 1),
        families=(family,),
        fan_in=False,
        weight_scale=2.0,
        mask_profile=profile,
        scale_exponent=exponent,
        dropout_profile="p1-beta" if profile == "p1-individual" else "fixed",
    )
    sampled = p.sample_network(
        req, named_rng=np.random.default_rng(16), raw_rng=np.random.default_rng(19)
    )
    sampled.validate_replay()
    probability = float(dict(sampled.choices)["dropout"])
    for i, block in enumerate(sampled.function.blocks):
        affine = block.experts[0]
        mask = np.array(affine.mask)
        if profile == "p1-individual":
            assert 0 <= probability <= 0.9
            if i == 0:
                assert mask.all()
            expected = 2 / (1 - (0 if i == 0 else probability) ** exponent)
        else:
            expected = 2 / float(mask.mean()) ** exponent
            # Each retained block is diagonal; the unequal-width remainder stays zero.
            assert mask.any()
        assert affine.multiplier == expected
        assert max(map(abs, affine.bias)) <= 1 / np.sqrt(affine.input_width)


def test_all_dropped_and_irregular_disconnected_draws_are_retained():
    for family in ("individual-sparse", "irregular"):
        request = p.NetworkSamplingRequest(
            2, (3, 1), families=(family,), dropout=1.0, skip_probability=0.0, bias_scale=0.0
        )
        sample = p.sample_network(
            request, named_rng=np.random.default_rng(3), raw_rng=np.random.default_rng(9)
        )
        value = p.evaluate_frozen_function(sample.function, torch.tensor([[1.0, 2.0], [-4.0, 7.0]]))
        assert torch.equal(value, torch.zeros_like(value))
        assert not any(v for b in sample.function.blocks for row in b.experts[0].mask for v in row)


def test_network_record_tamper_and_preallocation_limits():
    req = p.NetworkSamplingRequest(2, (3, 1), families=("residual",))
    sample = p.sample_network(
        req, named_rng=np.random.default_rng(4), raw_rng=np.random.default_rng(5)
    )
    task = p.LinearRecipe(rows=16, context_rows=8, features=2, signal_features=2, active_features=2)
    raw = json.loads(json.dumps(asdict(p.NetworkRecipe(task, sample))))
    raw["network"]["function"]["blocks"][0]["experts"][0]["bias"][0] += 1
    with pytest.raises(ValueError, match="replay"):
        p.recipe_from_dict(raw)
    raw = json.loads(json.dumps(asdict(p.NetworkRecipe(task, sample))))
    raw["network"]["raw"][0]["role"] = "wrong"
    with pytest.raises(ValueError, match="replay"):
        p.recipe_from_dict(raw)
    with pytest.raises(ValueError, match="limit"):
        replace(req, raw_limits=p.RawDrawLimits(numbers=1))
    with pytest.raises(ValueError, match="limit"):
        p.recipe_from_dict(
            asdict(p.NetworkRecipe(task, sample)),
            limits=replace(p.CausalLimits(), working_bytes=1024),
        )


def test_network_sampling_alias_and_failure_preserve_both_streams():
    import copy

    req = p.NetworkSamplingRequest(2, (3, 1))
    rng = np.random.default_rng(3)
    before = copy.deepcopy(rng.bit_generator.state)
    with pytest.raises(ValueError, match="distinct|alias|state|byte"):
        p.sample_network(req, named_rng=rng, raw_rng=rng)
    assert rng.bit_generator.state == before
    # State capacity failure occurs inside the atomic staging before publication.
    limited = replace(req, raw_limits=replace(req.raw_limits, state_bytes=1))
    one, two = np.random.default_rng(3), np.random.default_rng(4)
    states = copy.deepcopy((one.bit_generator.state, two.bit_generator.state))
    with pytest.raises(ValueError, match="distinct|alias|state|byte"):
        p.sample_network(limited, named_rng=one, raw_rng=two)
    assert (one.bit_generator.state, two.bit_generator.state) == states


def test_network_function_in_causal_native_parent_and_intervention():
    affine = p.NetworkAffine(((2.0, -1.0),), (0.5,))
    f = p.FrozenNetworkFunction(2, (p.NetworkBlock((0,), (affine,)),))
    parent = p.FrozenParentFunction((f,), aggregation="sum")
    assert p.frozen_function_from_dict(asdict(parent)) == parent
    root = p.FunctionalCausalNode(
        (), p.FrozenLinearFunction(((1.0, 0.0), (0.0, 1.0))), p.JointRootBase("gaussian", 2)
    )
    node = p.FunctionalCausalNode((0,), f)
    task = p.LinearRecipe(rows=16, context_rows=8, features=2, signal_features=2, active_features=2)
    causal = p.FunctionalCausalRecipe(task, (root, node), ((0, 0), (0, 1)), ((1, 0),))
    native = p.NativeCausalRecipe(
        task, (p.NativeCausalNode(root), p.NativeCausalNode(node)), ((0, 0), (0, 1)), (1, 0)
    )
    for base in (causal, native):
        pair = p.generate_causal_pair(
            base, interventions=(p.HardIntervention(0, (0, 1), (3.0, 4.0)),), seed=6, task_id=2
        )
        assert torch.all(pair.after.targets == 2.5)
        for task in (pair.before, pair.after):
            inspection = p.explain_generation(task, [15, 0, 15])
            assert any("expert" in n["id"] for n in inspection["nodes"])
            identifiers = {n["id"] for n in inspection["nodes"]}
            assert all(
                parent in identifiers for n in inspection["nodes"] for parent in n["parents"]
            )


@pytest.mark.parametrize("placement", ["before", "after"])
def test_activation_placement_has_literal_distinct_equations(placement: str):
    f = p.FrozenNetworkFunction(
        2,
        (
            p.NetworkBlock(
                (0,),
                (
                    p.NetworkAffine(
                        ((2.0, -1.0),),
                        (0.4,),
                        activation=p.CausalActivation("tanh"),
                        placement=placement,
                    ),
                ),
            ),
        ),
    )
    x = np.array([[-1.0, 2.0], [0.2, -0.4]])
    expected = (
        np.tanh(x) @ np.array([[2.0], [-1.0]]) + 0.4
        if placement == "before"
        else np.tanh(x @ np.array([[2.0], [-1.0]]) + 0.4)
    )
    np.testing.assert_allclose(p.evaluate_frozen_function(f, torch.tensor(x)), expected, atol=1e-14)


def test_p1_beta_is_bound_to_retained_raw_gamma_draws():
    req = p.NetworkSamplingRequest(
        2, (3, 1), families=("individual-sparse",), dropout_profile="p1-beta"
    )
    sample = p.sample_network(
        req, named_rng=np.random.default_rng(11), raw_rng=np.random.default_rng(12)
    )
    raw = {r.role: r for r in sample.raw}
    a, b = float(raw["dropout-a"].array()), float(raw["dropout-b"].array())
    assert 0.1 <= a < 5
    assert 0.1 <= b < 5
    assert dict(raw["dropout-gamma-a"].parameters)["concentration"] == a
    assert dict(raw["dropout-gamma-b"].parameters)["concentration"] == b
    ga, gb = float(raw["dropout-gamma-a"].array()), float(raw["dropout-gamma-b"].array())
    assert float(dict(sample.choices)["dropout"]) == 0.9 * ga / (ga + gb)


def test_network_generation_archive_roundtrip(tmp_path: Path):
    from r0.modules.generation import load_group, save_group

    sample = p.sample_network(
        p.NetworkSamplingRequest(2, (3, 1), families=("gated",)),
        named_rng=np.random.default_rng(10),
        raw_rng=np.random.default_rng(11),
    )
    controls = p.LinearRecipe(
        rows=16, context_rows=8, features=2, signal_features=2, active_features=2
    )
    task = p.generate_task(p.NetworkRecipe(controls, sample), seed=3, task_id=4)
    path = tmp_path / "network.zip"
    save_group(path, [(4, task)], max_bytes=900000)
    loaded = next(load_group(path))
    assert loaded.episode() == task.episode()
    assert p.explain_generation(loaded, [0, 15, 0])["signal_max_error"] == 0


def test_inspection_links_final_pre_activation_gate_and_skip_affines():
    expert = p.NetworkAffine(((1.0, 2.0),), (0.0,))
    gate = p.NetworkAffine(((1.0, 0.0), (0.0, 1.0)), (0.0, 0.0), placement="before")
    skip = p.NetworkProjection(0, p.NetworkAffine(((3.0, 4.0),), (1.0,), placement="before"))
    network = p.FrozenNetworkFunction(2, (p.NetworkBlock((0,), (expert, expert), gate, (skip,)),))
    controls = p.LinearRecipe(
        rows=16, context_rows=8, features=2, signal_features=2, active_features=2
    )
    task = p.generate_task(p.NetworkRecipe(controls, network), seed=3, task_id=4)
    nodes = {n["id"]: n for n in p.explain_generation(task, [0, 1])["nodes"]}
    assert nodes["network-block-1-weights"]["parents"] == ["network-block-1-gate-affine"]
    assert "network-block-1-skip-0-affine" in nodes["network-block-1-state"]["parents"]


def test_hidden_gate_width_is_admitted_before_evaluation():
    expert = p.NetworkAffine(((1.0,),), (0.0,))
    gate = p.NetworkAffine(((1.0,),) * 5, (0.0,) * 5)
    network = p.FrozenNetworkFunction(1, (p.NetworkBlock((0,), (expert,) * 5, gate),))
    with pytest.raises(ValueError, match="channel limit"):
        p.evaluate_frozen_function(
            network, torch.ones((2, 1)), limits=replace(p.CausalLimits(), node_channels=4)
        )
