"""Independent noise placement, importance and source custody oracles."""

import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pytest
import torch

from r0.modules import priors as p


def source(placement: str = "after"):
    function = p.FrozenNetworkFunction(
        2,
        (
            p.NetworkBlock(
                (0,),
                (
                    p.NetworkAffine(
                        ((2.0, -1.0),),
                        (0.3,),
                        activation=p.CausalActivation("tanh"),
                        placement=placement,
                    ),
                ),
            ),
        ),
    )
    task = p.LinearRecipe(rows=16, context_rows=8, features=2, signal_features=2, active_features=2)
    return p.NetworkRecipe(task, function)


@pytest.mark.parametrize("placement", ["post-block", "pre-activation"])
def test_authored_noise_and_replay(placement: str):
    recipe = p.with_network_innovations(
        source(),
        p.NetworkInnovationPlan(
            (p.NetworkInnovationStage((0.2,), (0.4,), placement),), importance=(3.0, 0.5)
        ),
    )
    task = p.generate_task(recipe, seed=7, task_id=9007199254740995)
    decoded = p.recipe_from_dict(json.loads(task.recipe_json)["recipe"])
    assert decoded == recipe
    assert torch.equal(
        task.targets, p.generate_task(decoded, seed=7, task_id=9007199254740995).targets
    )
    trace = p.explain_generation(task, [15, 0, 15])
    assert any("expert" in n["id"] for n in trace["nodes"])
    payload = asdict(recipe)
    payload["plan"]["importance"] = (4.0, 0.5)
    with pytest.raises(ValueError, match="runtime differs"):
        p.recipe_from_dict(payload)


def test_importance_is_outgoing_weight_not_input_rescale():
    recipe = p.with_network_innovations(
        source("before"), p.NetworkInnovationPlan((None,), importance=(3.0, 0.5))
    )
    task = p.generate_task(recipe, seed=7, task_id=3)
    x = task.inputs.numpy()
    expected = np.tanh(x) @ np.array([6.0, -0.5]) + 0.3
    np.testing.assert_allclose(task.scores.numpy()[:, 0], expected, rtol=1e-12, atol=1e-12)
    assert not np.allclose(expected, np.tanh(x * [3.0, 0.5]) @ [2.0, -1.0] + 0.3)


def test_sampled_noise_statistics_replay_and_rollback():
    request = p.NetworkInnovationRequest(
        p.NetworkScalarPrior("uniform", 0.1, 0.5),
        mean=p.NetworkScalarPrior("uniform", -0.2, 0.2),
        importance=p.NetworkScalarPrior("choice", choices=(0.0, 0.2, 3.0)),
        channel_scale="half-normal",
    )
    rng = np.random.default_rng(11)
    recipe = p.sample_network_innovations(source(), request, statistics_rng=rng)
    assert recipe.sampling is not None
    assert recipe.source is not None
    recipe.sampling.validate_replay(recipe.source)
    decoded = p.recipe_from_dict(asdict(recipe))
    assert decoded == recipe


@pytest.mark.parametrize("placement", ["pre-activation", "post-block"])
def test_fixed_innovation_response_numpy(placement: str):
    recipe = p.with_network_innovations(
        source(), p.NetworkInnovationPlan((p.NetworkInnovationStage((0.2,), (0.4,), placement),))
    )
    x = torch.tensor([[0.2, -0.8], [-1.0, 0.9]], dtype=torch.float64)
    z = torch.tensor([[1.7], [-0.6]], dtype=torch.float64)
    affine = x.numpy() @ [2.0, -1.0] + 0.3
    expected = (
        np.tanh(affine + 0.2 + 0.4 * z.numpy()[:, 0])
        if placement == "pre-activation"
        else np.tanh(affine) + 0.2 + 0.4 * z.numpy()[:, 0]
    )
    np.testing.assert_allclose(
        p.evaluate_network_innovation_reference(recipe, x, innovations={2: z}).numpy()[:, 0],
        expected,
        rtol=1e-13,
    )
    zero = np.tanh(affine + 0.2) if placement == "pre-activation" else np.tanh(affine) + 0.2
    np.testing.assert_allclose(
        p.evaluate_network_innovation_reference(recipe, x).numpy()[:, 0], zero, rtol=1e-13
    )


@pytest.mark.parametrize(
    "family",
    ["dense", "individual-sparse", "block-sparse", "skip", "residual", "irregular", "gated"],
)
def test_all_architectures_zero_noise_and_matched_interventions(family: str):
    sampled = p.sample_network(
        p.NetworkSamplingRequest(2, (3, 2, 1), families=(family,)),
        named_rng=np.random.default_rng(3),
        raw_rng=np.random.default_rng(5),
    )
    original = p.NetworkRecipe(source().task, sampled)
    plan = p.NetworkInnovationPlan(
        tuple(
            p.NetworkInnovationStage((0.0,) * b.output_width, (0.0,) * b.output_width)
            for b in original.function.blocks
        )
    )
    recipe = p.with_network_innovations(original, plan)
    before = p.generate_task(recipe, seed=17, task_id=3)
    plain = p.generate_task(original, seed=17, task_id=3)
    torch.testing.assert_close(before.inputs, plain.inputs, rtol=0, atol=0)
    torch.testing.assert_close(before.scores, plain.scores, rtol=0, atol=0)
    pair = p.generate_causal_pair(
        recipe,
        interventions=(p.HardIntervention(recipe.block_nodes[-1], (0,), (2.5,)),),
        seed=17,
        task_id=3,
    )
    assert torch.all(pair.after.targets == 2.5)
    assert all(json.loads(pair.card_json)["unaffected_exogenous_equal"].values())
    graph = p.explain_generation(before, [15, 1, 15])
    available: set[str] = set()
    for node in graph["nodes"]:
        assert set(node["parents"]) <= available
        available.add(node["id"])


def test_pinned_first_affine_exemption_and_internal_keyed_noise():
    first = p.NetworkAffine(((2.0, -1.0),), (0.3,), placement="before")
    second = p.NetworkAffine(
        ((3.0,),), (-0.2,), activation=p.CausalActivation("tanh"), placement="before"
    )
    original = p.NetworkRecipe(
        source().task,
        p.FrozenNetworkFunction(
            2, (p.NetworkBlock((0,), (first,)), p.NetworkBlock((1,), (second,)))
        ),
    )
    request = p.network_source_innovations(
        source="tabicl-9530bd6",
        scale=p.NetworkScalarPrior(low=0.4, high=0.4),
        pre_sample_scales=True,
    )
    recipe = p.sample_network_innovations(
        original, request, statistics_rng=np.random.default_rng(2)
    )
    assert recipe.plan is not None
    assert recipe.plan.stages[0] is None
    before = p.generate_task(recipe, seed=17, task_id=3)
    pair = p.generate_causal_pair(
        recipe,
        interventions=(p.HardIntervention(recipe.block_nodes[1], (0,), (0.7,)),),
        seed=17,
        task_id=3,
    )
    x = before.inputs.numpy()
    original_signal = 3 * np.tanh(x @ [2.0, -1.0] + 0.3) - 0.2
    noise = before.targets.numpy() - original_signal
    np.testing.assert_allclose(
        pair.after.targets.numpy(), 3 * np.tanh(0.7) - 0.2 + noise, rtol=1e-12, atol=1e-12
    )
    assert p.recipe_from_dict(asdict(recipe)) == recipe


def test_statistics_failure_rollback_and_raw_tamper():
    import copy
    from dataclasses import replace

    rng = np.random.default_rng(17)
    initial = copy.deepcopy(rng.bit_generator.state)
    bad = p.NetworkInnovationRequest(p.NetworkScalarPrior("uniform", -2.0, -1.0))
    with pytest.raises(ValueError, match="negative"):
        p.sample_network_innovations(source(), bad, statistics_rng=rng)
    assert rng.bit_generator.state == initial
    bad = replace(bad, raw_limits=p.RawDrawLimits(draws=1))
    with pytest.raises(ValueError, match="before draws"):
        p.sample_network_innovations(source(), bad, statistics_rng=rng)
    assert rng.bit_generator.state == initial
    good = replace(
        bad, scale=p.NetworkScalarPrior("uniform", 0.1, 0.5), raw_limits=p.RawDrawLimits()
    )
    recipe = p.sample_network_innovations(source(), good, statistics_rng=rng)
    payload = asdict(recipe)
    payload["sampling"]["parameters"][0]["value"] += 0.1
    with pytest.raises(ValueError, match="replay"):
        p.recipe_from_dict(payload)


@pytest.mark.parametrize("law", ["gaussian", "uniform", "student_t", "mixture"])
def test_broad_exogenous_laws_and_prefix(law: str):
    from dataclasses import replace

    original = source()
    stage = p.NetworkInnovationStage(
        (0.1,),
        (0.3,),
        exogenous=p.ExogenousDistribution(p.InputDistribution(distribution=law, df=4.5)),
        response=p.NoiseResponse(offset=0.5, strength=0.2),
    )
    plan = p.NetworkInnovationPlan((stage,))
    short = p.with_network_innovations(original, plan)
    long = p.with_network_innovations(replace(original, task=replace(original.task, rows=32)), plan)
    a = p.generate_task(short, seed=17, task_id=3)
    b = p.generate_task(long, seed=17, task_id=3)
    torch.testing.assert_close(a.scores, b.scores[:16], rtol=0, atol=0)
    assert p.recipe_from_dict(asdict(short)) == short


def test_multiclass_save_reopen(tmp_path: Path):
    from dataclasses import replace

    from r0.modules.generation import load_group, save_group

    original = source()
    task = replace(original.task, task_type="multiclass", classes=3)
    f = p.FrozenNetworkFunction(
        2,
        (
            p.NetworkBlock(
                (0,), (p.NetworkAffine(((2.0, -1.0), (0.1, 0.3), (-0.5, 2.0)), (0.2, 0.1, -0.3)),)
            ),
        ),
    )
    plan = p.NetworkInnovationPlan(
        (
            p.NetworkInnovationStage(
                (0.1, -0.2, 0.4),
                (0.2, 0.4, 0.5),
                exogenous=p.ExogenousDistribution(
                    factor=((1.0, 0.2, 0.0), (0.0, 1.0, 0.3), (0.1, 0.0, 1.0))
                ),
            ),
        )
    )
    recipe = p.with_network_innovations(p.NetworkRecipe(task, f), plan)
    generated = p.generate_task(recipe, seed=4, task_id=9007199254740995)
    path = tmp_path / "noise.zip"
    save_group(path, [(9007199254740995, generated)], max_bytes=900000)
    loaded = next(load_group(path))
    torch.testing.assert_close(loaded.scores, generated.scores, rtol=0, atol=0)
    assert p.recipe_from_dict(json.loads(loaded.recipe_json)["recipe"]) == recipe


def test_hyperprior_architecture_and_statistics_custody_compose():
    request = p.NetworkHyperpriorRequest(
        p.NetworkSamplingRequest(2, (1,), families=("gated",)),
        depth=p.NetworkScalarPrior(low=1.0, high=1.0),
        width=p.NetworkScalarPrior(low=3.0, high=3.0),
        weight_scale=p.NetworkScalarPrior(low=0.2, high=0.2),
        max_hidden_blocks=1,
        max_hidden_width=3,
    )
    architecture = p.sample_hyperprior_network(
        request,
        hyper_rng=np.random.default_rng(1),
        named_rng=np.random.default_rng(2),
        raw_rng=np.random.default_rng(3),
    )
    original = p.NetworkRecipe(source().task, architecture)
    noise = p.NetworkInnovationRequest(
        p.NetworkScalarPrior("uniform", 0.1, 0.5),
        scale_scope="per-channel",
        mean_scope="per-block",
        importance=p.NetworkScalarPrior("log-uniform", 0.1, 3.0),
    )
    recipe = p.sample_network_innovations(original, noise, statistics_rng=np.random.default_rng(4))
    assert p.recipe_from_dict(asdict(recipe)) == recipe
    payload = asdict(recipe)
    payload["source"]["network"]["parameters"][0]["value"] += 1
    with pytest.raises(ValueError, match="replay"):
        p.recipe_from_dict(payload)
