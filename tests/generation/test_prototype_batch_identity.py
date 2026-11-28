"""Producer records must retain exact integer identities and complete request custody."""

import json
from dataclasses import replace
from pathlib import Path

import pytest

from r0.modules import priors as p
from r0.modules.generation import RunRequest, load_group, run_generation


def batch_request():
    task = p.LinearRecipe(rows=16, context_rows=8, features=1, signal_features=1, active_features=1)
    function = p.FrozenNetworkFunction(
        1, (p.NetworkBlock((0,), (p.NetworkAffine(((2.0,),), (1.0,)),)),)
    )
    return p.PrototypeBatchNetworkRecipe(p.NetworkRecipe(task, function), rho=0.5)


@pytest.mark.parametrize("field", ["seed", "task_id"])
@pytest.mark.parametrize("replacement", [1.0, True])
@pytest.mark.parametrize("location", ["outer", "inner"])
def test_batch_producer_rejects_noninteger_identities(
    tmp_path: Path, field: str, replacement: float | bool, location: str
):
    recipe = batch_request()

    def produce(task_id: int):
        good = p.generate_task(recipe, seed=1, task_id=task_id)
        record = json.loads(good.recipe_json)
        target = record if location == "outer" else record["recipe"]
        target[field] = replacement
        return replace(good, recipe_json=json.dumps(record, sort_keys=True, separators=(",", ":")))

    result = run_generation(
        RunRequest(recipe, seed=1, start=1, count=1, max_seconds=30, max_bytes=4_000_000),
        tmp_path,
        produce=produce,
    )
    assert result.completed == 0
    assert result.failed == 1
    for archive in tmp_path.rglob("data-*.zip"):
        with pytest.raises(ValueError, match="completion receipt"):
            list(load_group(archive))


@pytest.mark.parametrize("change", ["number-type", "missing-default"])
def test_batch_producer_preserves_exact_declared_request(tmp_path: Path, change: str):
    recipe = batch_request()

    def produce(task_id: int):
        good = p.generate_task(recipe, seed=5, task_id=task_id)
        record = json.loads(good.recipe_json)
        source = record["recipe"]["request"]["source"]
        if change == "number-type":
            source["network"]["blocks"][0]["experts"][0]["bias"][0] = 1
        else:
            del source["network"]["blocks"][0]["experts"][0]["placement"]
        return replace(good, recipe_json=json.dumps(record, sort_keys=True, separators=(",", ":")))

    result = run_generation(
        RunRequest(recipe, seed=5, count=1, max_seconds=30, max_bytes=4_000_000),
        tmp_path,
        produce=produce,
    )
    assert result.completed == 0
    assert result.failed == 1


def test_batch_large_integer_archive_replays_exactly(tmp_path: Path):
    seed, task_id = 2**54 + 1, 2**54 + 3
    result = run_generation(
        RunRequest(
            batch_request(), seed=seed, start=task_id, count=1, max_seconds=30, max_bytes=4_000_000
        ),
        tmp_path,
    )
    assert result.completed == 1
    saved = next(load_group(next(tmp_path.rglob("data-*.zip"))))
    record = json.loads(saved.recipe_json)
    for field, expected in (("seed", seed), ("task_id", task_id)):
        assert type(record[field]) is int
        assert type(record["recipe"][field]) is int
        assert record[field] == record["recipe"][field] == expected
    replay = p.generate_task(p.recipe_from_dict(record["recipe"]), seed=seed, task_id=task_id)
    assert replay.recipe_json == saved.recipe_json
    assert replay.episode() == saved.episode()
