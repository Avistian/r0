"""Interventions preserve independent target calibration instead of fitting changed rows."""

import json
from dataclasses import asdict
from pathlib import Path
from typing import Literal

import numpy as np
import pytest

from r0.modules import generation as g
from r0.modules import priors as p


def _base(
    family: str,
    task_type: Literal["regression", "binary", "multiclass"] = "regression",
    noise: float = 0.0,
):
    width = 3 if task_type == "multiclass" else 1
    matrix = tuple(tuple(float(i == j) for j in range(width)) for i in range(width))
    task = p.LinearRecipe(
        rows=16,
        context_rows=7,
        features=1,
        signal_features=1,
        active_features=1,
        noise_ratio=noise,
        task_type=task_type,
        classes={"regression": 0, "binary": 2, "multiclass": 3}[task_type],
    )
    limits = p.CausalLimits(parameters=8_000_000, working_bytes=2 * 1024**3)
    nodes = (
        p.FunctionalCausalNode(
            (), p.FrozenLinearFunction(matrix), p.JointRootBase("gaussian", width)
        ),
        p.FunctionalCausalNode((0,), p.FrozenLinearFunction(matrix)),
        p.FunctionalCausalNode(
            (1,), p.FrozenLinearFunction(tuple(tuple(3 * v for v in row) for row in matrix))
        ),
    )
    if family == "functional":
        base = p.FunctionalCausalRecipe(
            task, nodes, ((0, 0),), tuple((2, j) for j in range(width)), limits=limits
        )
    elif family == "native":
        base = p.NativeCausalRecipe(
            task, tuple(p.NativeCausalNode(n) for n in nodes), ((0, 0),), (2, 0), limits=limits
        )
    elif family == "converted":
        base = p.ConvertedCausalRecipe(
            task,
            tuple(p.ConvertedCausalNode(n) for n in nodes),
            (p.ConvertedReadout(0, channel=0),),
            p.ConvertedReadout(2, channel=0),
            limits=limits,
        )
    else:
        base = p.NativeVectorRecipe(
            task,
            tuple(p.NativeVectorNode(n, (p.VariableDomain(),)) for n in nodes),
            ((0, 0),),
            (2, 0),
            limits=limits,
        )
    plan = p.RecursiveForestPlan(
        input_width=width,
        intervals=((-1.0, 1.0),) * width,
        leaf_mean=(0.1,) * width,
        leaf_scale=(1.0,) * width,
        source="r0-extension-v1",
        estimators=1,
        min_depth=1,
        max_depth=1,
        lifecycle="fresh-forest-per-row",
        populations=16,
    )
    original = p.realize_recursive_population_graph(
        p.RecursivePopulationGraphRequest(
            base, (p.RecursiveNodePopulationPlan(1, plan),), limits=limits
        ),
        seed=37,
        task_id=2**55 + 9,
    )
    return p.calibrate_population_graph(
        original, calibration_rows=32, calibration_seed=71, calibration_task_id=73
    )


@pytest.mark.parametrize("family", ["functional", "native", "converted", "vector"])
@pytest.mark.parametrize("mode", ["matched", "population"])
def test_fixed_calibration_pair_archive_trace(family: str, mode: str, tmp_path: Path):
    base = _base(family)
    change = p.CausalInterventionRecipe(
        base.original.request.base,
        (p.SoftIntervention(2, function=p.FrozenLinearFunction(((6.0,),))),),
        mode=mode,
        population=7,
    )
    pair = p.generate_calibrated_population_pair(base, change)
    after_recipe = p.recipe_from_dict(
        json.loads(pair.after.recipe_json)["recipe"], limits=base.limits
    )
    assert isinstance(after_recipe, p.CalibratedPopulationIntervention)
    assert asdict(after_recipe.base) == asdict(base)
    raw = p.generate_task(after_recipe.original, seed=37, task_id=2**55 + 9)
    expected = (raw.scores.numpy() - np.array(base.calibration.center)) / np.array(
        base.calibration.scale
    )
    np.testing.assert_allclose(pair.after.scores, expected, atol=1e-14)
    if mode == "matched":
        np.testing.assert_allclose(
            pair.after.targets,
            2 * pair.before.targets.numpy()
            + np.array(base.calibration.center)[0] / base.calibration.scale[0],
        )
    card = p.compare_causal_pair(pair.before, pair.after, limits=base.limits)
    assert card["calibration_equal"] is True
    path = tmp_path / "pair.zip"
    g.save_causal_pair(path, pair, max_bytes=64 * 1024**2, limits=base.limits)
    restored = g.load_causal_pair(path, limits=base.limits)
    assert restored.after.recipe_json == pair.after.recipe_json
    trace = p.explain_generation(restored.after, [15, 0, 15], limits=base.limits)
    assert trace["target_max_error"] == 0
    seen: set[str] = set()
    for node in trace["nodes"]:
        assert set(node["parents"]) <= seen
        assert node["id"] not in seen
        seen.add(node["id"])


@pytest.fixture(scope="module")
def matched_pair():
    base = _base("functional")
    change = p.CausalInterventionRecipe(
        base.original.request.base, (p.HardIntervention(2, (0,), (2.5,)),)
    )
    return base, p.generate_calibrated_population_pair(base, change)


@pytest.mark.parametrize("mutation", ["rows", "mode", "identity", "calibration", "budget"])
def test_calibrated_pair_admission_precedes_restore(
    matched_pair: tuple[p.CalibratedPopulationGraph, p.TaskPair], mutation: str
):
    from dataclasses import replace
    from unittest.mock import patch

    base, pair = matched_pair
    raw = json.loads(pair.after.recipe_json)["recipe"]
    cap = base.limits
    if mutation == "rows":
        raw["original"]["base"]["request"]["base"]["task"]["rows"] = True
    elif mutation == "mode":
        raw["original"]["change"]["mode"] = "unknown"
    elif mutation == "identity":
        raw["original"]["base"]["seed"] += 1
    elif mutation == "calibration":
        raw["base"]["calibration"]["center"][0] = 0
    else:
        cap = replace(cap, parameters=1)
    cls = type(base.original.bindings[0].sample.initial)
    with (
        patch.object(cls, "restore", side_effect=AssertionError("metadata reached restore")),
        patch.object(np.random, "default_rng", side_effect=AssertionError("metadata created RNG")),
        pytest.raises(ValueError, match="calibrat|population|limit|metadata|intervention"),
    ):
        p.recipe_from_dict(raw, limits=cap)


@pytest.mark.parametrize("side", ["before", "after"])
@pytest.mark.parametrize("mutation", ["task-type", "score-dtype", "seed-type"])
def test_calibrated_pair_full_task_custody(
    matched_pair: tuple[p.CalibratedPopulationGraph, p.TaskPair], side: str, mutation: str
):
    from dataclasses import replace

    base, pair = matched_pair
    task = getattr(pair, side)
    if mutation == "task-type":
        forged = replace(task, task_type="binary")
    elif mutation == "score-dtype":
        forged = replace(task, scores=task.scores.float())
    else:
        record = json.loads(task.recipe_json)
        record["seed"] = float(record["seed"])
        forged = replace(task, recipe_json=json.dumps(record))
    with pytest.raises(ValueError, match="population|identity|identities|semantic"):
        p.compare_causal_pair(
            forged if side == "before" else pair.before,
            forged if side == "after" else pair.after,
            limits=base.limits,
        )


@pytest.mark.parametrize("task_type", ["regression", "binary", "multiclass"])
@pytest.mark.parametrize("mode", ["matched", "population"])
def test_calibrated_changed_adapter_reuses_correct_streams(
    task_type: Literal["regression", "binary", "multiclass"], mode: str
):
    base = _base("functional", task_type, 0.35 if task_type == "regression" else 0.0)
    width = 3 if task_type == "multiclass" else 1
    change = p.CausalInterventionRecipe(
        base.original.request.base,
        (p.HardIntervention(2, tuple(range(width)), tuple(1.5 + j for j in range(width))),),
        mode=mode,
        population=5,
    )
    pair = p.generate_calibrated_population_pair(base, change)
    expected = (
        np.array([1.5 + j for j in range(width)]) - base.calibration.center
    ) / base.calibration.scale
    np.testing.assert_allclose(pair.after.scores, np.broadcast_to(expected, (16, width)))
    if task_type == "regression":
        residual = pair.after.targets.numpy() - pair.after.scores[:, 0].numpy()
        previous = pair.before.targets.numpy() - pair.before.scores[:, 0].numpy()
        if mode == "matched":
            np.testing.assert_allclose(residual, previous, atol=1e-14)
        else:
            assert not np.array_equal(residual, previous)
    else:
        assert pair.after.probabilities is not None
        if task_type == "binary":
            prob = 1 / (1 + np.exp(-expected[0] * base.task.separation))
            probabilities = np.tile([1 - prob, prob], (16, 1))
        else:
            logits = expected * base.task.separation
            exp = np.exp(logits - logits.max())
            probabilities = np.tile(exp / exp.sum(), (16, 1))
        np.testing.assert_allclose(pair.after.probabilities, probabilities, atol=1e-14)
    trace = p.explain_generation(pair.after, [15, 0, 15], limits=base.limits)
    assert trace["target_max_error"] == 0


def test_pair_manifest_boolean_custody(
    matched_pair: tuple[p.CalibratedPopulationGraph, p.TaskPair], tmp_path: Path
):
    from dataclasses import replace

    base, pair = matched_pair
    card = json.loads(pair.card_json)
    card["calibration_equal"] = 1
    with pytest.raises(ValueError, match="card differs"):
        g.save_causal_pair(
            tmp_path / "bad",
            replace(pair, card_json=json.dumps(card)),
            max_bytes=64 * 1024**2,
            limits=base.limits,
        )
    assert not (tmp_path / "bad").exists()
