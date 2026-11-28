"""Public vector intervention arithmetic and replay custody."""

import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import torch

from r0.modules import priors as p
from r0.modules.generation import load_causal_pair, save_causal_pair


def authored() -> p.NativeVectorRecipe:
    continuous = p.VariableDomain()
    binary = p.VariableDomain("binary", ("no", "yes"), "threshold", (0.0,))
    categorical = p.VariableDomain("categorical", ("a", "b", "c"), "softmax")
    root = p.NativeVectorNode(
        p.FunctionalCausalNode(
            (),
            p.FrozenLinearFunction(((1.0,), (1.0,), (-1.0,), (0.0,), (1.0,))),
            p.JointRootBase("gaussian", 1),
        ),
        (continuous, binary, categorical),
    )
    child = p.NativeVectorNode(
        p.FunctionalCausalNode(
            (0,),
            p.FrozenLinearFunction(
                ((1.0, 0.0, 10.0, 0.0, 1.0, 2.0), (1.0, 0.0, 10.0, 0.0, 1.0, 2.0))
            ),
        ),
        (continuous, binary),
    )
    return p.NativeVectorRecipe(
        p.LinearRecipe(rows=32, context_rows=16, features=4, signal_features=4, active_features=4),
        (root, child),
        ((0, 0), (0, 1), (0, 2), (1, 1)),
        (1, 0),
    )


def test_partial_clamp_preserves_joint_siblings_and_encoded_descendants(tmp_path: Path) -> None:
    base = authored()
    pair = p.generate_causal_pair(
        base,
        interventions=(p.NativeVectorHardIntervention(0, 2, category="c"),),
        seed=17,
        task_id=5,
    )
    np.testing.assert_array_equal(pair.after.inputs[:, :2], pair.before.inputs[:, :2])
    np.testing.assert_array_equal(pair.after.inputs[:, 2], 2)
    np.testing.assert_array_equal(
        pair.after.targets, pair.after.inputs[:, 0] + 10 * pair.after.inputs[:, 1] + 2
    )
    decoded = p.recipe_from_dict(json.loads(pair.after.recipe_json)["recipe"])
    assert isinstance(decoded, p.CausalInterventionRecipe)
    assert decoded.base == base
    assert all(json.loads(pair.card_json)["unaffected_exogenous_equal"].values())
    path = tmp_path / "pair"
    save_causal_pair(path, pair, max_bytes=10_000_000)
    restored = load_causal_pair(path)
    assert restored.after.recipe_json == pair.after.recipe_json
    assert p.explain_generation(pair.after, [2, 0, 2])["rows"] == [2, 0, 2]


def test_full_clamp_bypasses_joint_draws_and_threshold_score_span() -> None:
    base = authored()
    changes = (
        p.NativeVectorHardIntervention(0, 0, value=3.0),
        p.NativeVectorHardIntervention(0, 1, category="yes"),
        p.NativeVectorHardIntervention(0, 2, category="b"),
    )
    pair = p.generate_causal_pair(base, interventions=changes, seed=17, task_id=5)
    np.testing.assert_array_equal(pair.after.targets, 14)
    card = json.loads(pair.after.mechanism_json)["intervention"]
    assert card["bypassed_nodes"] == [0]
    assert not any("root" in key for key in card["stream_initial_fingerprints"])
    graph = p.explain_generation(pair.after, [0])
    score = next(n for n in graph["nodes"] if n["id"] == "intervention-0-score")
    assert score["shape"] == [1, 5]


def test_domain_stage_population_and_rejections() -> None:
    base = authored()
    changed = p.FrequencyVariableDomain(
        "categorical", ("a", "b", "c"), "softmax", frequencies=(0.1, 0.2, 0.7), temperature=2.0
    )
    pair = p.generate_causal_pair(
        base, interventions=(p.NativeVectorDomainIntervention(0, 2, changed),), seed=17, task_id=5
    )
    checks = json.loads(pair.card_json)["unaffected_exogenous_equal"]
    assert checks["0:uniform:2"]
    fresh = p.generate_causal_pair(
        base,
        interventions=(p.NativeVectorHardIntervention(0, 2, category="a"),),
        mode="population",
        seed=17,
        task_id=5,
    )
    assert torch.equal(fresh.before.context_indices, fresh.after.context_indices)
    assert not torch.equal(fresh.before.inputs[:, 0], fresh.after.inputs[:, 0])
    for change in (
        p.NativeVectorHardIntervention(0, 3, value=0.0),
        p.NativeVectorHardIntervention(0, 1, category="bad"),
        p.HardIntervention(0, (0,), (1.0,)),
        p.NativeVectorDomainIntervention(
            0, 1, p.VariableDomain("binary", ("no", "yes"), "softmax")
        ),
    ):
        with pytest.raises(ValueError, match="vector|variable|vocabulary|width"):
            p.CausalInterventionRecipe(base, (change,))


def test_correlated_innovation_and_combined_joint_stage_changes() -> None:
    base = authored()
    root = base.nodes[0]
    width = root.equation.function.output_width
    factor = tuple(tuple(1.0 if j <= i else 0.0 for j in range(width)) for i in range(width))
    noisy = replace(
        base,
        nodes=(
            replace(
                root,
                equation=replace(
                    root.equation,
                    noise_scale=(0.2,) * width,
                    exogenous=p.ExogenousDistribution(factor=factor),
                ),
            ),
            *base.nodes[1:],
        ),
    )
    changes = (
        p.SoftIntervention(0, noise_scale=(0.4,) * width),
        p.StageIntervention(0, root=p.JointRootBase("uniform", 1)),
        p.NativeVectorHardIntervention(0, 2, category="c"),
        p.NativeVectorDomainIntervention(
            0, 1, p.VariableDomain("binary", ("no", "yes"), "threshold", (0.3,))
        ),
    )
    pair = p.generate_causal_pair(noisy, interventions=changes, seed=17, task_id=5)
    card = json.loads(pair.card_json)
    assert card["unaffected_exogenous_equal"]["0:exogenous"]
    assert "0:root" not in card["unaffected_exogenous_equal"]
    assert all(card["stream_initial_equal"].values())
    np.testing.assert_array_equal(
        pair.after.targets, pair.after.inputs[:, 0] + 10 * pair.after.inputs[:, 1] + 2
    )
    graph = p.explain_generation(pair.after, [0, 1, 2])
    by_id = {n["id"]: np.asarray(n["values"]) for n in graph["nodes"]}
    np.testing.assert_allclose(
        by_id["intervention-0-score"][:, 0],
        by_id["intervention-0-function"][:, 0] + 0.4 * by_id["intervention-0-exogenous"][:, 0],
    )
    with pytest.raises(ValueError, match="overlapping"):
        p.CausalInterventionRecipe(
            base,
            (
                p.SoftIntervention(0, noise_scale=(0.2,) * width),
                p.SoftIntervention(0, noise_scale=(0.3,) * width),
            ),
        )


def test_full_clamp_avoids_overflowing_function_and_partial_does_not() -> None:
    base = authored()
    bad = replace(
        base,
        nodes=(
            replace(
                base.nodes[0],
                equation=replace(
                    base.nodes[0].equation, function=p.FrozenLinearFunction(((1e308,),) * 5)
                ),
            ),
            *base.nodes[1:],
        ),
    )
    all_clamps = (
        p.NativeVectorHardIntervention(0, 0, value=3.0),
        p.NativeVectorHardIntervention(0, 1, category="yes"),
        p.NativeVectorHardIntervention(0, 2, category="c"),
    )
    # Direct after-generation bypasses a mechanism that need not admit finite row outputs.
    after = p.generate_task(p.CausalInterventionRecipe(bad, all_clamps), seed=17, task_id=5)
    np.testing.assert_array_equal(after.targets, 15)
    with pytest.raises(ValueError, match="nonfinite"):
        p.generate_task(p.CausalInterventionRecipe(bad, (all_clamps[0],)), seed=17, task_id=5)


def test_target_domain_probability_oracle_and_tamper_admission() -> None:
    base = authored()
    base = replace(
        base,
        task=replace(
            base.task,
            features=2,
            signal_features=2,
            active_features=2,
            task_type="multiclass",
            classes=3,
        ),
        nodes=(base.nodes[0],),
        observed=((0, 0), (0, 1)),
        target=(0, 2),
    )
    domain = p.FrequencyVariableDomain(
        "categorical", ("a", "b", "c"), "softmax", frequencies=(0.1, 0.2, 0.7), temperature=2.0
    )
    pair = p.generate_causal_pair(
        base, interventions=(p.NativeVectorDomainIntervention(0, 2, domain),), seed=17, task_id=5
    )
    x = pair.after.inputs[:, 0].numpy()
    weights = np.exp(np.stack((-x, np.zeros_like(x), x), axis=1) / 2) * np.array([0.1, 0.2, 0.7])
    assert pair.after.probabilities is not None
    np.testing.assert_allclose(
        pair.after.probabilities, weights / weights.sum(axis=1, keepdims=True), rtol=1e-14
    )
    with pytest.raises(ValueError, match="metadata"):
        p.compare_causal_pair(pair.before, replace(pair.after, mechanism_json="{}"))
    with pytest.raises(ValueError, match="arrays"):
        p.compare_causal_pair(
            pair.before, replace(pair.after, targets=(pair.after.targets + 1) % 3)
        )
    limit = base.admit()["estimated_working_bytes"] + 1000
    tight = replace(base, limits=replace(base.limits, working_bytes=limit))
    with pytest.raises(ValueError, match="working memory"):
        p.CausalInterventionRecipe(tight, (p.NativeVectorHardIntervention(0, 2, category="a"),))


def sampled_request(parent: bool = False, function: str = "lin") -> p.NativeVectorGraphRequest:
    c = p.NativeDomainPrior(kinds=("continuous",))
    b = p.NativeDomainPrior(
        kinds=("binary",),
        emissions=("threshold",),
        frequency_laws=("uniform",),
        threshold_laws=("uniform-quantile",),
    )
    k = p.AsymmetricNativeDomainPrior(
        kinds=("categorical",),
        cardinalities=(3,),
        emissions=("softmax",),
        concentration_vectors=((1.0, 2.0, 4.0),),
    )
    k5 = p.AsymmetricNativeDomainPrior(
        kinds=("categorical",),
        cardinalities=(5,),
        emissions=("softmax",),
        concentration_vectors=((1.0, 2.0, 3.0, 4.0, 5.0),),
    )
    f = p.JointFunctionPrior(functions=function, matrix=p.JointMatrixPrior(families=("gaussian",)))
    return p.NativeVectorGraphRequest(
        parents=((), (0,)),
        domains=((c, b, k), (k5, c, b)),
        observed=((0, 0), (0, 1), (0, 2), (1, 1)),
        target=(1, 0),
        rows=16,
        context_rows=8,
        calibration_rows=8,
        root_prior=p.JointRootPrior(families=("gaussian",), function=f),
        function_prior=f,
        composition=p.ParentCompositionPrior(
            source="extension", mode="aggregate", aggregation="sum"
        )
        if parent
        else None,
    )


def sample(request: p.NativeVectorGraphRequest):
    return p.sample_native_vector_graph(
        request,
        **{
            s: np.random.default_rng(i)
            for i, s in enumerate(("named_rng", "raw_rng", "calibration_rng", "fit_rng"))
        },
    )


@pytest.mark.parametrize("parent", [False, True])
def test_sampled_vector_parent_retains_construction_archive(parent: bool) -> None:
    graph = sample(sampled_request(parent))
    pair = p.generate_causal_pair(
        graph,
        interventions=(
            p.NativeVectorHardIntervention(
                1, 0, category=graph.recipe.nodes[1].domains[0].vocabulary[-1]
            ),
        ),
        seed=17,
        task_id=5,
    )
    before = json.loads(pair.before.recipe_json)["recipe"]
    after = json.loads(pair.after.recipe_json)["recipe"]
    assert after["base"] == before
    assert torch.all(pair.after.targets == 4)
    decoded = p.recipe_from_dict(after)
    assert isinstance(decoded, p.CausalInterventionRecipe)
    assert decoded.base == graph


def test_joint_pipeline_replace_and_remove() -> None:
    base = authored()
    piped = replace(
        base,
        nodes=(
            replace(
                base.nodes[0],
                equation=replace(
                    base.nodes[0].equation, pipeline=p.CausalNodePipeline(5, node_importance=3.0)
                ),
            ),
            *base.nodes[1:],
        ),
    )
    removed = p.generate_causal_pair(
        piped, interventions=(p.StageIntervention(0, replace_pipeline=True),), seed=17, task_id=5
    )
    np.testing.assert_allclose(removed.before.inputs[:, 0], 3 * removed.after.inputs[:, 0])
    changed = p.generate_causal_pair(
        piped,
        interventions=(
            p.StageIntervention(
                0, replace_pipeline=True, pipeline=p.CausalNodePipeline(5, node_importance=2.0)
            ),
        ),
        seed=17,
        task_id=5,
    )
    np.testing.assert_allclose(changed.after.inputs[:, 0], 2 * removed.after.inputs[:, 0])
