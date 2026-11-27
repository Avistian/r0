"""Explicit batch requests reconstruct populations; realized archives retain every row."""

import json
from dataclasses import asdict, replace
from pathlib import Path
from typing import Literal
from unittest.mock import patch

import numpy as np
import pytest
import torch

from r0.modules import priors as p


def request(
    mode: str = "multinomial-trial-temperature-r0",
    task_type: Literal["regression", "multiclass"] = "regression",
):
    outputs = 3 if task_type == "multiclass" else 1
    task = p.LinearRecipe(
        rows=16,
        context_rows=8,
        features=2,
        signal_features=2,
        active_features=2,
        task_type=task_type,
        classes=outputs if outputs > 1 else 0,
    )
    net = p.FrozenNetworkFunction(
        2,
        (
            p.NetworkBlock(
                (0,),
                (
                    p.NetworkAffine(
                        tuple((float(i + 2), 3.0) for i in range(outputs)), (1.0,) * outputs
                    ),
                ),
            ),
        ),
    )
    return p.PrototypeBatchNetworkRecipe(
        p.NetworkRecipe(task, net),
        rho=0.25,
        weights=p.PrototypeWeightLaw(
            mode=mode, trials=None if mode == "dirichlet-concentration-r0" else 16
        ),
    )


@pytest.mark.parametrize(
    "mode",
    [
        "multinomial-trial-temperature-r0",
        "multinomial-probability-temperature-r0",
        "dirichlet-concentration-r0",
    ],
)
@pytest.mark.parametrize("task_type", ["regression", "multiclass"])
def test_batch_generation_literal_numpy_custody_and_repeated_rows(
    mode: str, task_type: Literal["regression", "multiclass"]
):
    recipe = request(mode, task_type)
    generated = p.generate_task(recipe, seed=19, task_id=2**54 + 3)
    restored = p.recipe_from_dict(json.loads(generated.recipe_json)["recipe"])
    assert isinstance(restored, p.RealizedPrototypeBatchNetworkRecipe)
    population = restored.population
    bank_rng, weights_rng = (s.restore() for s in population.initial)
    bank = bank_rng.standard_normal((4, 2))
    if mode == "dirichlet-concentration-r0":
        raw = weights_rng.gamma([1.0] * 4, size=(16, 4))
        weights = raw / raw.sum(axis=1, keepdims=True)
    else:
        raw = weights_rng.multinomial(16, [0.25] * 4, size=16)
        weights = raw / 16
    np.testing.assert_array_equal(population.root.prototypes, bank)
    np.testing.assert_array_equal(population.raw[-1].array(), raw)
    np.testing.assert_array_equal(generated.inputs, weights @ bank)
    expected = (
        generated.inputs.numpy()
        @ np.array([[2.0, 3.0], [3.0, 3.0], [4.0, 3.0]][: generated.scores.shape[1]]).T
        + 1
    )
    np.testing.assert_allclose(generated.scores, expected, rtol=1e-14, atol=1e-14)
    replay = p.generate_task(restored, seed=19, task_id=2**54 + 3)
    assert replay.recipe_json == generated.recipe_json
    assert torch.equal(replay.targets, generated.targets)
    trace = p.explain_generation(generated, [15, 0, 15])
    assert trace["rows"] == [15, 0, 15]
    by_id = {node["id"]: node for node in trace["nodes"]}
    np.testing.assert_array_equal(by_id["prototype-weights"]["values"], weights[[15, 0, 15]])
    preceding: set[str] = set()
    for node in trace["nodes"]:
        assert set(node["parents"]) <= preceding
        preceding.add(node["id"])


def test_new_population_identity_and_row_count_reconstruct_banks_and_weights():
    first = p.realize_prototype_batch_network(request(), seed=1, task_id=3)
    next_task = p.generate_task(first, seed=1, task_id=4)
    second = p.recipe_from_dict(json.loads(next_task.recipe_json)["recipe"])
    assert isinstance(second, p.RealizedPrototypeBatchNetworkRecipe)
    assert first.population.root != second.population.root
    larger = first.with_task(replace(first.task, rows=32))
    assert isinstance(larger, p.PrototypeBatchNetworkRecipe)
    population = p.realize_prototype_batch_network(larger, seed=1, task_id=3)
    assert population.population.root.count == 8
    assert len(population.population.draw.values) == 32
    changed_split = first.with_task(replace(first.task, context_rows=6, split=1))
    assert isinstance(changed_split, p.RealizedPrototypeBatchNetworkRecipe)
    assert changed_split.population == first.population
    a = p.generate_task(first, seed=1, task_id=3)
    b = p.generate_task(changed_split, seed=1, task_id=3)
    assert a.source_task_id == b.source_task_id
    assert a.episode_id != b.episode_id
    with pytest.raises(ValueError, match="identity|identities"):
        p.generate_task(first, seed=True, task_id=3)


@pytest.mark.parametrize("key", ["seed", "task_id", "request", "population", "version"])
def test_realized_archive_identity_and_required_fields_fail_closed(key: str):
    realized = p.realize_prototype_batch_network(request(), seed=1, task_id=3)
    raw = asdict(realized)
    del raw[key]
    with pytest.raises(ValueError, match="prototype|batch"):
        p.recipe_from_dict(raw)
    raw = asdict(realized)
    raw["seed"] = 2
    with pytest.raises(ValueError, match="identity"):
        p.recipe_from_dict(raw)


def test_known_raw_and_workspace_limits_reject_before_population_rng_copy():
    source = request().source
    with (
        patch("copy.deepcopy", side_effect=AssertionError("premature RNG clone")),
        pytest.raises(ValueError, match="raw.*limit"),
    ):
        p.PrototypeBatchNetworkRecipe(source, raw_limits=p.RawDrawLimits(numbers=1))
    with pytest.raises(ValueError, match="limit|memory"):
        p.PrototypeBatchNetworkRecipe(replace(source, limits=p.CausalLimits(working_bytes=1000)))


def test_task_noise_pair_retains_exact_population_and_archive_controls():
    pair = p.generate_pair(request(), intervention="noise", value=0.5, seed=11, task_id=3)
    before, after = (
        p.recipe_from_dict(json.loads(t.recipe_json)["recipe"]) for t in (pair.before, pair.after)
    )
    assert isinstance(before, p.RealizedPrototypeBatchNetworkRecipe)
    assert isinstance(after, p.RealizedPrototypeBatchNetworkRecipe)
    assert before.population == after.population
    assert torch.equal(pair.before.inputs, pair.after.inputs)
    assert torch.equal(pair.before.scores, pair.after.scores)
    assert json.loads(pair.card_json)["independent_source_draws"] == 1
    with pytest.raises(ValueError, match="noise/separation"):
        p.paired_recipe(before, intervention="irrelevant", value=3)


def test_batch_source_profile_normalization_and_strict_nested_fields():
    profile = p.sample_source_root_profile(
        p.SourceRootProfile(2, selector="forced-r0", families=("categorical", "zipf_clipped")),
        model_rng=np.random.default_rng(1),
        selection_rng=np.random.default_rng(2),
    )
    source_root = p.sample_source_root_base(
        profile,
        modes=("center", "standardize"),
        calibration_rows=32,
        calibration_rng=np.random.default_rng(3),
    )
    recipe = replace(request(), source=replace(request().source, root=source_root))
    realized = p.realize_prototype_batch_network(recipe, seed=5, task_id=7)
    assert p.recipe_from_dict(asdict(realized)) == realized
    emitted = p.generate_task(realized, seed=5, task_id=7)
    assert p.explain_generation(emitted, [15, 0, 15])["signal_max_error"] == 0
    for key in asdict(recipe.source.task):
        raw = asdict(recipe)
        del raw["source"]["task"][key]
        with pytest.raises(ValueError, match="."):
            p.prototype_batch_network_from_dict(raw)


def test_batch_population_seed_and_limits_cannot_be_relabelled():
    realized = p.realize_prototype_batch_network(request(), seed=5, task_id=7)
    with pytest.raises(ValueError, match="identity"):
        replace(realized, task_id=8)
    higher = replace(realized.population, limits=p.CausalLimits(working_bytes=512 * 1024**2))
    with pytest.raises(ValueError, match="limits"):
        replace(realized, population=higher)


def test_batch_run_archives_reopen_realized_populations(tmp_path: Path):
    from r0.modules.generation import RunRequest, load_group, run_generation

    summary = run_generation(
        RunRequest(
            request(), seed=5, start=2, count=2, group_size=2, max_seconds=20, max_bytes=8_000_000
        ),
        tmp_path,
    )
    assert summary.completed == 2
    tasks = [
        task for archive in sorted(tmp_path.rglob("data-*.zip")) for task in load_group(archive)
    ]
    assert len(tasks) == 2
    roots: list[p.FrozenPrototypeRoot] = []
    for task in tasks:
        record = json.loads(task.recipe_json)
        recipe = p.recipe_from_dict(record["recipe"])
        assert isinstance(recipe, p.RealizedPrototypeBatchNetworkRecipe)
        roots.append(recipe.population.root)
        replay = p.generate_task(recipe, seed=record["seed"], task_id=record["task_id"])
        assert replay.episode() == task.episode()
        assert p.explain_generation(task, [15, 0, 15])["rows"] == [15, 0, 15]
    assert roots[0] != roots[1]


@pytest.mark.parametrize("mismatch", ["request", "identity"])
def test_batch_runner_rejects_valid_but_unrequested_realization(tmp_path: Path, mismatch: str):
    from r0.modules.generation import RunRequest, run_generation

    original = request()
    altered = replace(original, rho=0.5) if mismatch == "request" else original

    def produce(task_id: int):
        return p.generate_task(altered, seed=6 if mismatch == "identity" else 5, task_id=task_id)

    summary = run_generation(
        RunRequest(original, seed=5, count=1, max_seconds=20, max_bytes=4_000_000),
        tmp_path,
        produce=produce,
    )
    assert summary.failed == 1
    assert summary.completed == 0
    failures = next(tmp_path.rglob("failures-*.jsonl")).read_text()
    assert "different task or recipe" in failures


def test_batch_request_missing_version_cannot_fall_back_to_linear():
    raw = asdict(request())
    del raw["version"]
    with pytest.raises(ValueError, match="prototype batch.*version"):
        p.recipe_from_dict(raw)
