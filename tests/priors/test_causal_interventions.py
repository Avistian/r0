"""Literal structural equations, intervention stages and persisted original-row replay."""

import json
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

import pytest
import torch

from r0.modules import priors as p
from r0.modules.generation import load_causal_pair, save_causal_pair


def fork():
    task = p.LinearRecipe(rows=16, context_rows=8, features=2, signal_features=2, active_features=2)
    return p.FunctionalCausalRecipe(
        task,
        (
            p.FunctionalCausalNode(
                (), p.FrozenLinearFunction(((1.0,),)), p.JointRootBase("gaussian", 1)
            ),
            p.FunctionalCausalNode((0,), p.FrozenLinearFunction(((2.0,),)), noise_scale=(0.5,)),
            p.FunctionalCausalNode(
                (0, 1), p.FrozenLinearFunction(((-1.0, 3.0),)), noise_scale=(0.7,)
            ),
        ),
        ((0, 0), (1, 0)),
        ((2, 0),),
    )


def test_hard_and_soft_literal_fork_and_matched_innovations():
    base = fork()
    before = p.generate_task(base, seed=17, task_id=3)
    a, b = before.inputs.T
    uc = before.targets - (-a + 3 * b)
    hard = p.generate_causal_pair(
        base, interventions=(p.HardIntervention(1, (0,), (4.0,)),), seed=17, task_id=3
    )
    torch.testing.assert_close(hard.after.targets, -a + 12 + uc)
    torch.testing.assert_close(hard.after.inputs[:, 0], a, rtol=0, atol=0)
    assert torch.all(hard.after.inputs[:, 1] == 4)
    assert all(json.loads(hard.card_json)["unaffected_exogenous_equal"].values())
    soft = p.generate_causal_pair(
        base,
        interventions=(p.SoftIntervention(1, function=p.FrozenLinearFunction(((-1.0,),))),),
        seed=17,
        task_id=3,
    )
    expected_b = -a + (b - 2 * a)
    torch.testing.assert_close(soft.after.inputs[:, 1], expected_b)
    torch.testing.assert_close(soft.after.targets, -a + 3 * expected_b + uc)


@pytest.mark.parametrize("location", ["state", "readout", "state_and_readout"])
def test_pipeline_locations_have_distinct_downstream_implications(location: str):
    base = fork()
    pipe = p.CausalNodePipeline(1, converters=(p.NumericalConverter(0),), node_importance=3.0)
    base = replace(
        base, nodes=(base.nodes[0], replace(base.nodes[1], pipeline=pipe), base.nodes[2])
    )
    pair = p.generate_causal_pair(
        base,
        interventions=(p.HardIntervention(1, (0,), (4.0,), location=location),),
        seed=17,
        task_id=3,
    )
    if location == "state":
        torch.testing.assert_close(pair.before.inputs, pair.after.inputs, rtol=0, atol=0)
    else:
        assert torch.all(pair.after.inputs[:, 1] == 4)
    if location == "readout":
        torch.testing.assert_close(pair.before.targets, pair.after.targets, rtol=0, atol=0)
    else:
        assert not torch.equal(pair.before.targets, pair.after.targets)
    assert p.explain_generation(pair.after, [3, 0, 3])["rows"] == [3, 0, 3]


def test_vector_channel_and_multiple_interventions():
    base = fork()
    nodes = (
        base.nodes[0],
        p.FunctionalCausalNode((0,), p.FrozenLinearFunction(((2.0,), (-3.0,)))),
        p.FunctionalCausalNode((1,), p.FrozenLinearFunction(((1.0, 2.0),))),
    )
    base = replace(base, nodes=nodes, observed=((1, 0), (1, 1)))
    pair = p.generate_causal_pair(
        base, interventions=(p.HardIntervention(1, (0,), (4.0,)),), seed=7, task_id=2
    )
    torch.testing.assert_close(pair.before.inputs[:, 1], pair.after.inputs[:, 1], rtol=0, atol=0)
    torch.testing.assert_close(pair.after.targets, 4 + 2 * pair.before.inputs[:, 1])
    both = p.generate_causal_pair(
        base,
        interventions=(p.HardIntervention(0, (0,), (1.0,)), p.HardIntervention(1, (0,), (4.0,))),
        seed=7,
        task_id=2,
    )
    assert torch.all(both.after.targets == -2)


def test_whole_replacement_bypasses_overflowing_overwritten_function():
    base = fork()
    base = replace(
        base,
        nodes=(
            base.nodes[0],
            replace(base.nodes[1], function=p.FrozenLinearFunction(((1e308,),))),
            base.nodes[2],
        ),
    )
    # Standalone surgical generation can replace an equation even when its observational run fails.
    recipe = p.CausalInterventionRecipe(base, (p.HardIntervention(1, (0,), (4.0,)),))
    result = p.generate_task(recipe, seed=17, task_id=3)
    assert torch.isfinite(result.targets).all()
    assert 1 in json.loads(result.mechanism_json)["intervention"]["bypassed_nodes"]


def test_population_reuses_mechanisms_not_exogenous_units():
    pair = p.generate_causal_pair(
        fork(),
        interventions=(p.HardIntervention(1, (0,), (4.0,)),),
        mode="population",
        population=9,
        seed=17,
        task_id=3,
    )
    card = json.loads(pair.card_json)
    assert card["split_equal"]
    assert not all(card["unaffected_exogenous_equal"].values())
    assert not torch.equal(pair.before.inputs[:, 0], pair.after.inputs[:, 0])


def native():
    return p.NativeCausalRecipe(
        p.LinearRecipe(
            rows=16,
            context_rows=8,
            features=1,
            signal_features=1,
            active_features=1,
            task_type="binary",
            classes=2,
        ),
        (
            p.NativeCausalNode(
                p.FunctionalCausalNode(
                    (), p.FrozenLinearFunction(((1.0,), (-1.0,))), p.JointRootBase("gaussian", 1)
                ),
                p.VariableDomain("categorical", ("red", "blue"), "softmax"),
            ),
            p.NativeCausalNode(
                p.FunctionalCausalNode((0,), p.FrozenLinearFunction(((2.0, -2.0), (-2.0, 2.0)))),
                p.VariableDomain("binary", ("no", "yes"), "softmax"),
            ),
        ),
        ((0, 0),),
        (1, 0),
    )


def test_native_category_do_updates_descendants_and_forced_probabilities():
    base = native()
    pair = p.generate_causal_pair(
        base, interventions=(p.HardIntervention(0, category="blue"),), seed=3, task_id=4
    )
    assert torch.all(pair.after.inputs == 1)
    forced = p.generate_causal_pair(
        base, interventions=(p.HardIntervention(1, category="yes"),), seed=3, task_id=4
    )
    assert torch.all(forced.after.targets == 1)
    assert forced.after.probabilities is not None
    assert torch.all(forced.after.probabilities[:, 1] == 1)
    assert torch.equal(forced.after.scores, forced.after.probabilities)
    assert p.recipe_from_dict(
        asdict(p.CausalInterventionRecipe(base, (p.HardIntervention(1, category="yes"),)))
    )


def test_archive_pair_and_inspection_roundtrip(tmp_path: Path):
    pair = p.generate_causal_pair(
        fork(), interventions=(p.HardIntervention(1, (0,), (4.0,)),), seed=17, task_id=3
    )
    save_causal_pair(tmp_path / "pair", pair, max_bytes=2_000_000)
    loaded = load_causal_pair(tmp_path / "pair")
    assert loaded.after.episode() == pair.after.episode()
    assert json.loads(loaded.card_json) == json.loads(pair.card_json)
    assert p.explain_generation(loaded.after, [15, 0, 15])["rows"] == [15, 0, 15]
    with pytest.raises(FileExistsError):
        save_causal_pair(tmp_path / "pair", pair, max_bytes=2_000_000)
    manifest = tmp_path / "pair" / "pair.json"
    raw = json.loads(manifest.read_text())
    raw["card"]["mode"] = "population"
    manifest.write_text(json.dumps(raw))
    with pytest.raises(
        ValueError, match="intervention|comparison|causal|channel|geometry|category|record|object"
    ):
        load_causal_pair(tmp_path / "pair")


@pytest.mark.parametrize(
    "changes",
    [
        (p.HardIntervention(99, (0,), (1.0,)),),
        (p.HardIntervention(1, (9,), (1.0,)),),
        (p.HardIntervention(1, (0,), (1.0,)), p.HardIntervention(1, (0,), (2.0,))),
        (p.HardIntervention(1, (0,), (1.0,)), p.SoftIntervention(1, noise_scale=(2.0,))),
        (p.SoftIntervention(1, function=p.FrozenLinearFunction(((1.0, 2.0),))),),
    ],
)
def test_invalid_multi_interventions_reject(
    changes: tuple[p.HardIntervention | p.SoftIntervention, ...],
):
    with pytest.raises(
        ValueError, match="intervention|comparison|causal|channel|geometry|category|record|object"
    ):
        p.CausalInterventionRecipe(fork(), changes)


def test_target_calibration_is_original_and_query_independent():
    original = fork()
    base = p.CalibratedFunctionalCausalRecipe(
        original.task, original.nodes, original.observed, original.target, calibration_rows=32
    )
    changes = (p.SoftIntervention(1, function=p.FrozenLinearFunction(((-1.0,),))),)
    pair = p.generate_causal_pair(base, interventions=changes, seed=17, task_id=3)
    assert (
        json.loads(pair.before.mechanism_json)["calibration"]
        == json.loads(pair.after.mechanism_json)["calibration"]
    )
    larger = replace(base, task=replace(base.task, rows=32, context_rows=4))
    another = p.generate_causal_pair(larger, interventions=changes, seed=17, task_id=3)
    assert (
        json.loads(another.after.mechanism_json)["calibration"]
        == json.loads(pair.after.mechanism_json)["calibration"]
    )


def test_pair_rejects_tampered_arrays_metadata_and_decoder():
    pair = p.generate_causal_pair(
        fork(), interventions=(p.HardIntervention(1, (0,), (4.0,)),), seed=17, task_id=3
    )
    with pytest.raises(
        ValueError, match="intervention|comparison|causal|channel|geometry|category|record|object"
    ):
        p.compare_causal_pair(pair.before, replace(pair.after, scores=pair.after.scores + 1))
    with pytest.raises(
        ValueError, match="intervention|comparison|causal|channel|geometry|category|record|object"
    ):
        p.compare_causal_pair(pair.before, replace(pair.after, mechanism_json="{}"))
    raw = json.loads(pair.after.recipe_json)["recipe"]
    malformed: tuple[Any, ...] = (None, [], {"version": "wrong"})
    for value in malformed:
        broken = {**raw, "interventions": [value]}
        with pytest.raises(
            ValueError,
            match="intervention|comparison|causal|channel|geometry|category|record|object",
        ):
            p.recipe_from_dict(broken)


def test_native_category_validation_and_soft_geometry():
    with pytest.raises(
        ValueError, match="intervention|comparison|causal|channel|geometry|category|record|object"
    ):
        p.CausalInterventionRecipe(native(), (p.HardIntervention(0, category="missing"),))
    with pytest.raises(
        ValueError, match="intervention|comparison|causal|channel|geometry|category|record|object"
    ):
        p.CausalInterventionRecipe(native(), (p.HardIntervention(0, (0,), (1.0,)),))
    change = p.SoftIntervention(1, function=p.FrozenLinearFunction(((-2.0, 2.0), (2.0, -2.0))))
    pair = p.generate_causal_pair(native(), interventions=(change,), seed=3, task_id=4)
    assert all(json.loads(pair.card_json)["unaffected_exogenous_equal"].values())
    assert pair.before.probabilities is not None
    assert pair.after.probabilities is not None
    assert not torch.equal(pair.before.probabilities, pair.after.probabilities)


def test_readout_and_state_targets_are_order_independent():
    changes = (
        p.HardIntervention(1, (0,), (4.0,)),
        p.HardIntervention(1, (0,), (9.0,), location="readout"),
    )
    one = p.generate_causal_pair(fork(), interventions=changes, seed=7, task_id=2)
    two = p.generate_causal_pair(fork(), interventions=tuple(reversed(changes)), seed=7, task_id=2)
    torch.testing.assert_close(one.after.inputs, two.after.inputs, rtol=0, atol=0)
    torch.testing.assert_close(one.after.targets, two.after.targets, rtol=0, atol=0)
    assert torch.all(one.after.inputs[:, 1] == 9)


def test_comparison_admission_rejects_before_arrays():
    base = fork()
    declared = base.admit()["estimated_working_bytes"]
    limited = replace(base, limits=replace(base.limits, working_bytes=declared + 1))
    with pytest.raises(ValueError, match="comparison working memory limit"):
        p.CausalInterventionRecipe(limited, (p.HardIntervention(1, (0,), (4.0,)),))


def test_sampled_native_archive_retains_original_construction(tmp_path: Path):
    import numpy as np

    f = p.JointFunctionPrior(functions="lin", matrix=p.JointMatrixPrior(families=("gaussian",)))
    d = p.NativeDomainPrior(kinds=("categorical",), cardinalities=(3,), emissions=("softmax",))
    request = p.NativeParentGraphRequest(
        parents=((), (), (1, 0)),
        domains=(d,) * 3,
        observed=((0, 0), (1, 0)),
        target=(2, 0),
        rows=16,
        context_rows=8,
        calibration_rows=8,
        root_prior=p.JointRootPrior(families=("gaussian",), function=f),
        function_prior=f,
        composition=p.ParentCompositionPrior(
            source="extension", mode="aggregate", aggregation="sum"
        ),
    )
    streams = dict(
        zip(
            ("named_rng", "raw_rng", "calibration_rng", "fit_rng"),
            (np.random.default_rng(10 + i) for i in range(4)),
            strict=True,
        )
    )
    graph = p.sample_native_graph(request, **streams)
    label = graph.recipe.nodes[0].domain.vocabulary[1]
    pair = p.generate_causal_pair(
        graph, interventions=(p.HardIntervention(0, category=label),), seed=3, task_id=4
    )
    decoded = p.recipe_from_dict(json.loads(pair.after.recipe_json)["recipe"])
    assert isinstance(decoded, p.CausalInterventionRecipe)
    assert decoded.base == graph
    save_causal_pair(tmp_path / "native", pair, max_bytes=3_000_000)
    loaded = load_causal_pair(tmp_path / "native")
    assert loaded.after.episode() == pair.after.episode()
    assert p.explain_generation(loaded.after, [2, 0])["rows"] == [2, 0]


def test_native_continuous_clamp_bypasses_original_equation_and_importance():
    equation = p.FunctionalCausalNode(
        (),
        p.FrozenLinearFunction(((1e308,),)),
        p.JointRootBase("gaussian", 1),
        pipeline=p.CausalNodePipeline(1, node_importance=3.0),
    )
    task = p.LinearRecipe(rows=16, context_rows=8, features=1, signal_features=1, active_features=1)
    continuous = p.NativeCausalRecipe(
        task,
        (
            p.NativeCausalNode(equation, p.VariableDomain()),
            p.NativeCausalNode(
                p.FunctionalCausalNode((0,), p.FrozenLinearFunction(((2.0,),))), p.VariableDomain()
            ),
        ),
        ((0, 0),),
        (1, 0),
    )
    result = p.generate_task(
        p.CausalInterventionRecipe(continuous, (p.HardIntervention(0, (0,), (4.0,)),)),
        seed=17,
        task_id=3,
    )
    assert torch.all(result.inputs == 4)
    assert torch.all(result.targets == 8)
