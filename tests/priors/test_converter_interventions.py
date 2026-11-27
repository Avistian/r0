"""Literal category cuts distinguish capture, pre-noise representation and final state."""

import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import torch

from r0.modules import priors as p
from r0.modules.generation import load_causal_pair, save_causal_pair


def graph(mode: str = "neighbor_disc", *, noise: float = 0.2) -> p.ConvertedCausalRecipe:
    converter = p.CategoricalConverter(
        "kind",
        0,
        2,
        ("left", "right"),
        mode,
        ((-2.0, -3.0), (4.0, 5.0)),
        mean=(0.0, 0.0) if mode.startswith("softmax") else (),
        scale=(1.0, 1.0) if mode.startswith("softmax") else (),
        bias=(0.0, 0.0) if mode.startswith("softmax") else (),
        post_function=p.FrozenLinearFunction(((2.0, 0.0), (0.0, 3.0)))
        if mode == "neighbor_func"
        else None,
    )
    root = p.ConvertedCausalNode(
        p.FunctionalCausalNode(
            (),
            p.FrozenLinearFunction(((1.0,), (-1.0,))),
            p.JointRootBase("gaussian", 1),
            noise_scale=(noise, noise),
            pipeline=p.CausalNodePipeline(2, node_importance=2.0),
        ),
        (converter,),
    )
    child = p.ConvertedCausalNode(
        p.FunctionalCausalNode(
            (0,),
            p.FrozenLinearFunction(((2.0, 1.0),)),
        )
    )
    return p.ConvertedCausalRecipe(
        p.LinearRecipe(rows=16, context_rows=8, features=1, signal_features=1, active_features=1),
        (root, child),
        (p.ConvertedReadout(0, capture="kind"),),
        p.ConvertedReadout(1, channel=0),
    )


@pytest.mark.parametrize(
    "mode",
    [
        "neighbor_id",
        "neighbor_disc",
        "neighbor_func",
        "neighbor_int",
        "softmax_id",
        "softmax_disc",
        "softmax_int",
    ],
)
def test_capture_only_preserves_descendants_and_replays(mode: str, tmp_path: Path) -> None:
    base = graph(mode)
    pair = p.generate_causal_pair(
        base, interventions=(p.ConverterHardIntervention(0, "kind", "right"),), seed=23, task_id=5
    )
    assert torch.equal(pair.before.targets, pair.after.targets)
    assert torch.equal(pair.after.inputs, torch.ones_like(pair.after.inputs))
    card = json.loads(pair.card_json)
    assert all(card["unaffected_exogenous_equal"].values())
    path = tmp_path / "pair"
    save_causal_pair(path, pair, max_bytes=5_000_000)
    restored = load_causal_pair(path)
    assert restored.after.recipe_json == pair.after.recipe_json
    trace = p.explain_generation(pair.after, [15, 0, 15])
    assert trace["rows"] == [15, 0, 15]
    seen: set[str] = set()
    for item in trace["nodes"]:
        assert set(item["parents"]).issubset(seen)
        seen.add(item["id"])
    capture = next(n for n in trace["nodes"] if n["id"] == "intervention-0-scores:kind")
    assert capture["parents"] == []


@pytest.mark.parametrize(
    ("mode", "representation"),
    [
        ("neighbor_disc", (4.0, 5.0)),
        ("neighbor_func", (8.0, 15.0)),
        ("neighbor_int", (1.0, 1.0)),
        ("softmax_disc", (4.0, 5.0)),
        ("softmax_int", (1.0, 1.0)),
    ],
)
def test_representation_cut_retains_noise_and_importance(
    mode: str, representation: tuple[float, float]
) -> None:
    base = graph(mode)
    pair = p.generate_causal_pair(
        base,
        interventions=(
            p.ConverterHardIntervention(0, "kind", "right", "representation_and_capture"),
        ),
        seed=23,
        task_id=5,
    )
    before = p.explain_generation(pair.before, [*range(16)])
    raw = np.asarray(next(n for n in before["nodes"] if n["id"] == "exogenous-0")["values"])
    expected = 2 * (np.asarray(representation) + 0.2 * raw) @ np.asarray([2.0, 1.0])
    np.testing.assert_allclose(pair.after.targets.numpy(), expected, rtol=1e-14)
    metadata = json.loads(pair.after.mechanism_json)["intervention"]
    assert metadata["converter_semantics"]["stage_bypassed_nodes"] == [0]
    trace = p.explain_generation(pair.after, [1, 1])
    assert not any(n["id"] == "intervention-0-function" for n in trace["nodes"])


def test_final_state_cut_preserves_capture_and_soft_replacement_is_frozen() -> None:
    base = graph()
    pair = p.generate_causal_pair(
        base, interventions=(p.HardIntervention(0, (0, 1), (3.0, 4.0)),), seed=4, task_id=2
    )
    assert torch.equal(pair.before.inputs, pair.after.inputs)
    np.testing.assert_array_equal(pair.after.targets.numpy(), 10.0)
    converter = replace(base.nodes[0].converters[0], representatives=((-4.0, -6.0), (8.0, 10.0)))
    changes = (p.ConverterSoftIntervention(0, "kind", converter),)
    soft = p.generate_causal_pair(base, interventions=changes, seed=4, task_id=2)
    expected = p.generate_task(
        replace(base, nodes=(replace(base.nodes[0], converters=(converter,)), base.nodes[1])),
        seed=4,
        task_id=2,
    )
    assert torch.equal(soft.after.targets, expected.targets)
    decoded = p.recipe_from_dict(json.loads(soft.after.recipe_json)["recipe"])
    assert isinstance(decoded, p.CausalInterventionRecipe)
    assert decoded.base == base


def test_bypass_replaced_overflow_and_reject_ambiguous_locations() -> None:
    base = graph(noise=0.0)
    bad = replace(
        base.nodes[0],
        equation=replace(
            base.nodes[0].equation, function=p.FrozenLinearFunction(((1e308,), (1e308,)))
        ),
    )
    base = replace(base, nodes=(bad, base.nodes[1]))
    recipe = p.CausalInterventionRecipe(
        base, (p.ConverterHardIntervention(0, "kind", "right", "representation_and_capture"),)
    )
    result = p.generate_task(recipe, seed=23, task_id=5)
    np.testing.assert_array_equal(result.targets.numpy(), 26.0)
    for changes in (
        (p.ConverterHardIntervention(0, "missing", "right"),),
        (p.ConverterHardIntervention(0, "kind", "missing"),),
        (p.ConverterHardIntervention(0, "kind", "right"),) * 2,
        (p.HardIntervention(0, (0,), (1.0,), location="readout"),),
    ):
        with pytest.raises(ValueError, match="capture|vocabulary|overlapping|converter"):
            p.CausalInterventionRecipe(graph(), changes)
    with pytest.raises(ValueError, match="unique"):
        p.CausalInterventionRecipe(
            graph("neighbor_id"),
            (p.ConverterHardIntervention(0, "kind", "right", "representation_and_capture"),),
        )


def test_direct_target_forced_scores_population_and_multiple_stage_changes(tmp_path: Path) -> None:
    original = graph("softmax_disc")
    base = replace(
        original,
        task=replace(original.task, task_type="binary", classes=2),
        observed=(p.ConvertedReadout(1, channel=0),),
        target=p.ConvertedReadout(0, capture="kind"),
    )
    changes = (p.ConverterHardIntervention(0, "kind", "right"), p.HardIntervention(1, (0,), (7.0,)))
    pair = p.generate_causal_pair(base, interventions=changes, seed=13, task_id=2)
    np.testing.assert_array_equal(pair.after.targets.numpy(), 1)
    np.testing.assert_array_equal(pair.after.scores.numpy(), np.tile([0.0, 1.0], (16, 1)))
    assert pair.after.probabilities is not None
    assert torch.equal(pair.after.probabilities, pair.after.scores)
    np.testing.assert_array_equal(pair.after.inputs.numpy(), 7)
    population = p.generate_causal_pair(
        base, interventions=(changes[0],), mode="population", population=3, seed=13, task_id=2
    )
    assert not torch.equal(population.after.inputs, pair.before.inputs)
    assert torch.equal(population.before.context_indices, population.after.context_indices)
    save_causal_pair(tmp_path / "binary", pair, max_bytes=5_000_000)
    assert load_causal_pair(tmp_path / "binary").after.classes == ("left", "right")
    record = json.loads(pair.after.recipe_json)["recipe"]
    for field, value in (("category", "absent"), ("capture", 3), ("location", "state")):
        bad = json.loads(json.dumps(record))
        bad["interventions"][0][field] = value
        with pytest.raises(ValueError, match="converter"):
            p.recipe_from_dict(bad)


def test_partial_and_signal_dependent_representation_cuts_still_evaluate_needed_equation() -> None:
    base = graph(noise=0.2)
    root = replace(
        base.nodes[0],
        equation=replace(
            base.nodes[0].equation, noise_response=p.NoiseResponse(strength=0.3, power=1.0)
        ),
    )
    base = replace(base, nodes=(root, base.nodes[1]))
    pair = p.generate_causal_pair(
        base,
        interventions=(
            p.ConverterHardIntervention(0, "kind", "right", "representation_and_capture"),
        ),
        seed=17,
        task_id=2,
    )
    before = p.explain_generation(pair.before, list(range(16)))
    values = {n["id"]: np.asarray(n["values"]) for n in before["nodes"]}
    expected = (
        2 * (np.array([4.0, 5.0]) + values["noise-scale-0"] * values["exogenous-0"]) @ [2.0, 1.0]
    )
    np.testing.assert_allclose(pair.after.targets.numpy(), expected, rtol=1e-14)
    assert (
        json.loads(pair.after.mechanism_json)["intervention"]["converter_semantics"][
            "stage_bypassed_nodes"
        ]
        == []
    )
    # A fully overwritten final state and capture bypass even a replaced nonfinite post-function.
    exploding = replace(
        root.converters[0],
        mode="neighbor_func",
        post_function=p.FrozenLinearFunction(((1e308, 1e308), (1e308, 1e308))),
    )
    base = replace(base, nodes=(replace(root, converters=(exploding,)), base.nodes[1]))
    recipe = p.CausalInterventionRecipe(
        base,
        (
            p.ConverterHardIntervention(0, "kind", "right"),
            p.HardIntervention(0, (0, 1), (3.0, 4.0)),
        ),
    )
    result = p.generate_task(recipe, seed=17, task_id=2)
    np.testing.assert_array_equal(result.targets.numpy(), 10.0)
    assert json.loads(result.mechanism_json)["intervention"]["bypassed_nodes"] == [0]
