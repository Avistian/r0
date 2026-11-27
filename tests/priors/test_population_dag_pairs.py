"""Shared causal pair persistence retains complete row-owned graph populations."""

import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from r0.modules import generation as g
from r0.modules import priors as p


def population_pair(mode: str = "matched") -> p.TaskPair:
    root = p.ConvertedCausalNode(
        p.FunctionalCausalNode(
            (), p.FrozenLinearFunction(((1.0, 0.0), (0.0, 1.0))), p.JointRootBase("gaussian", 2)
        )
    )
    converter = p.CategoricalConverter(
        "kind", 0, 1, ("low", "high"), "neighbor_disc", ((-2.0,), (3.0,))
    )
    child = p.ConvertedCausalNode(
        p.FunctionalCausalNode((0,), p.FrozenLinearFunction(((1.0, 0.0), (0.0, 1.0)))), (converter,)
    )
    base = p.ConvertedCausalRecipe(
        p.LinearRecipe(rows=16, context_rows=7, features=2, active_features=2, signal_features=2),
        (root, child),
        (p.ConvertedReadout(0, channel=0), p.ConvertedReadout(0, channel=1)),
        p.ConvertedReadout(1, channel=1),
    )
    plan = p.RecursiveForestPlan(
        input_width=2,
        estimators=2,
        min_depth=1,
        max_depth=1,
        leaf_mean=(1.0, -2.0),
        leaf_scale=(0.5, 2.0),
        source="r0-extension-v1",
        lifecycle="fresh-forest-per-row",
        populations=16,
    )
    recipe = p.realize_recursive_population_graph(
        p.RecursivePopulationGraphRequest(base, (p.RecursiveNodePopulationPlan(1, plan),)),
        seed=31,
        task_id=2**54 + 37,
        rng=np.random.default_rng(41),
    )
    change = p.CausalInterventionRecipe(
        base, (p.HardIntervention(0, (0, 1), (1.0, -2.0)),), mode=mode, population=3
    )
    return p.generate_population_graph_pair(recipe, change)


@pytest.mark.parametrize("mode", ["matched", "population"])
def test_shared_pair_zip_and_compare_roundtrip(tmp_path: Path, mode: str):
    pair = population_pair(mode)
    assert p.compare_causal_pair(pair.before, pair.after) == json.loads(pair.card_json)
    path = tmp_path / "pair"
    g.save_causal_pair(path, pair, max_bytes=64 * 1024**2)
    loaded = g.load_causal_pair(path)
    assert loaded.before.recipe_json == pair.before.recipe_json
    assert loaded.after.recipe_json == pair.after.recipe_json
    assert json.loads(loaded.card_json) == json.loads(pair.card_json)
    assert json.loads(loaded.after.recipe_json)["task_id"] == 2**54 + 37
    for task in (loaded.before, loaded.after):
        trace = p.explain_generation(task, [12, 1, 12])
        target = next(n for n in trace["nodes"] if n["id"] == "target")
        np.testing.assert_array_equal(np.asarray(target["values"])[:, 0], task.targets[[12, 1, 12]])


@pytest.fixture(scope="module")
def retained_pair() -> p.TaskPair:
    return population_pair()


@pytest.mark.parametrize("side", ["before", "after"])
@pytest.mark.parametrize(
    "mutation",
    [
        "score_dtype",
        "targets",
        "seed_alias",
        "missing_recipe",
        "implementation",
        "forest_leaf",
        "input_rows",
    ],
)
def test_malformed_pair_members_reject_before_persistence(
    tmp_path: Path, retained_pair: p.TaskPair, side: str, mutation: str
):
    task = getattr(retained_pair, side)
    if mutation == "score_dtype":
        task = replace(task, scores=task.scores.float())
    elif mutation == "targets":
        task = replace(task, targets=task.targets + 0.5)
    else:
        raw = json.loads(task.recipe_json)
        if mutation == "seed_alias":
            raw["seed"] = float(raw["seed"])
        elif mutation == "missing_recipe":
            del raw["recipe"]
        elif mutation == "forest_leaf":
            sample = (
                raw["recipe"]["bindings"][0]["sample"]
                if side == "before"
                else raw["recipe"]["samples"][0]
            )
            sample["forests"][0]["trees"][0]["nodes"][1]["value"][0] += 0.25
        elif mutation == "input_rows":
            rows = (
                raw["recipe"]["bindings"][0]["inputs"]
                if side == "before"
                else raw["recipe"]["inputs"][0][1]
            )
            rows[0][0] += 0.25
        else:
            raw["execution"]["implementation"] = "unrelated-v1"
        task = replace(task, recipe_json=json.dumps(raw))
    invalid = (
        replace(retained_pair, before=task)
        if side == "before"
        else replace(retained_pair, after=task)
    )
    path = tmp_path / "rejected"
    with pytest.raises(
        ValueError, match="population|identity|identities|semantic|recursive|replay"
    ):
        g.save_causal_pair(path, invalid, max_bytes=64 * 1024**2)
    assert not path.exists()


def test_pair_caller_limit_precedes_any_snapshot_restore(tmp_path: Path, retained_pair: p.TaskPair):
    from unittest.mock import patch

    recipe = p.recipe_from_dict(json.loads(retained_pair.before.recipe_json)["recipe"])
    assert isinstance(recipe, p.RecursivePopulationGraph)
    limits = replace(recipe.limits, working_bytes=32 * 1024)
    path = tmp_path / "pair"
    g.save_causal_pair(path, retained_pair, max_bytes=64 * 1024**2)
    with patch.object(
        type(recipe.bindings[0].sample.initial),
        "restore",
        side_effect=AssertionError("early restore"),
    ):
        with pytest.raises(ValueError, match="limit|cap|budget|reservation"):
            p.compare_causal_pair(retained_pair.before, retained_pair.after, limits=limits)
        with pytest.raises(ValueError, match="limit|cap|budget|reservation"):
            g.load_causal_pair(path, limits=limits)


@pytest.mark.parametrize("mode", ["matched", "population"])
def test_guarded_saved_pair_inspect_export_reopen(tmp_path: Path, mode: str):
    import zipfile

    pair = population_pair(mode)
    path = tmp_path / "pair"
    g.save_causal_pair(path, pair, max_bytes=64 * 1024**2)
    request = {
        "max_seconds": 60,
        "max_bytes": 64 * 1024**2,
        "pair_path": str(path),
        "indices": [12, 1, 12],
    }
    for action in ("open", "inspect", "export"):
        result = g.run_guarded_document({**request, "causal_action": action}, tmp_path / action)
        assert result.returncode == 0, result.stderr
        assert result.summary is not None
        assert result.summary["status"] == "COMPLETE"
        if action == "export":
            destination = tmp_path / "extracted"
            with zipfile.ZipFile(str(result.summary["download"])) as archive:
                archive.extractall(destination)
            restored = g.load_causal_pair(destination)
            assert restored.before.recipe_json == pair.before.recipe_json
            assert restored.after.recipe_json == pair.after.recipe_json


def test_pair_card_numeric_alias_and_manifest_alias_reject(
    tmp_path: Path, retained_pair: p.TaskPair
):
    raw = json.loads(retained_pair.card_json)
    raw["base_archive_unchanged"] = 1
    with pytest.raises(ValueError, match="card differs"):
        g.save_causal_pair(
            tmp_path / "bad",
            replace(retained_pair, card_json=json.dumps(raw)),
            max_bytes=64 * 1024**2,
        )
    path = tmp_path / "valid"
    g.save_causal_pair(path, retained_pair, max_bytes=64 * 1024**2)
    manifest = json.loads((path / "pair.json").read_text())
    manifest["card"]["base_archive_unchanged"] = 1
    (path / "pair.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="evidence differs"):
        g.load_causal_pair(path)


def test_invalid_pair_identity_precedes_restore(retained_pair: p.TaskPair):
    from unittest.mock import patch

    recipe = p.recipe_from_dict(json.loads(retained_pair.before.recipe_json)["recipe"])
    assert isinstance(recipe, p.RecursivePopulationGraph)
    raw = json.loads(retained_pair.after.recipe_json)
    raw["execution"]["implementation"] = "unrelated-v1"
    task = replace(retained_pair.after, recipe_json=json.dumps(raw))
    with (
        patch.object(
            type(recipe.bindings[0].sample.initial),
            "restore",
            side_effect=AssertionError("early restore"),
        ),
        pytest.raises(ValueError, match="implementation identity"),
    ):
        p.compare_causal_pair(retained_pair.before, task)
