"""Finite GP columns feed explicit root contexts without iid resampling."""

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from gp_root_fixture import gp_root_recipe

from r0.modules import priors as p


@pytest.mark.parametrize("family", ["functional", "native", "vector", "converted"])
def test_retained_gp_root_inputs_and_literal_descendant(family: str):
    recipe = gp_root_recipe(family)
    population = recipe.source
    decoded = p.recipe_from_dict(json.loads(json.dumps(asdict(recipe))))
    assert asdict(decoded) == asdict(recipe)
    actual = p.generate_task(recipe, seed=47, task_id=59)
    retained = np.asarray(population.draw.values)
    np.testing.assert_array_equal(actual.inputs, retained)
    np.testing.assert_allclose(actual.targets, 2 * retained[:, 0] - retained[:, 1], atol=1e-14)


@pytest.mark.parametrize("mutation", ["binding", "columns", "rows", "version", "source", "budget"])
def test_complete_gp_root_metadata_precedes_covariance(
    mutation: str, monkeypatch: pytest.MonkeyPatch
):
    record = asdict(gp_root_recipe("functional"))
    cap = p.CausalLimits()
    if mutation == "binding":
        record["bindings"][0]["node"] = 1
    elif mutation == "columns":
        record["bindings"][0]["columns"] = (0, 99)
    elif mutation == "rows":
        record["base"]["task"]["rows"] = 17
    elif mutation == "version":
        record["bindings"][0]["version"] = "invalid"
    elif mutation == "source":
        record["source"]["draw"]["values"] = ()
    else:
        cap = p.CausalLimits(working_bytes=1024)

    def forbidden(*args: object, **kwargs: object):
        pytest.fail("covariance replay started before complete GP root admission")

    monkeypatch.setattr(np.linalg, "cholesky", forbidden)
    with pytest.raises(ValueError, match="GP|limit|row|matrix|archive"):
        p.recipe_from_dict(record, limits=cap)


def test_shared_gp_column_preserves_cross_root_dependency_diagnostic():
    source = p.realize_gp_columns(p.GPColumnRequest(rows=16), seed=47, task_id=59)
    task = p.LinearRecipe(rows=16, context_rows=8, features=1, signal_features=1, active_features=1)
    root = p.FunctionalCausalNode(
        (), p.FrozenLinearFunction(((1.0,),)), p.JointRootBase("gaussian", 1)
    )
    child = p.FunctionalCausalNode((1,), p.FrozenLinearFunction(((3.0,),)))
    base = p.FunctionalCausalRecipe(task, (root, root, child), ((0, 0),), ((2, 0),))
    recipe = p.GPRootDAGRecipe(source, base, (p.GPRootBinding(0, (0,)), p.GPRootBinding(1, (0,))))
    actual = p.generate_task(recipe, seed=47, task_id=59)
    np.testing.assert_allclose(actual.targets, 3 * actual.inputs[:, 0], atol=1e-14)
    assert "disconnected_target" not in json.loads(actual.mechanism_json)["runtime"]["flags"]


@pytest.mark.parametrize("columns", [(1, 0), (0, 0)])
def test_gp_columns_can_be_reordered_or_shared_within_one_root(columns: tuple[int, int]):
    from dataclasses import replace

    original = gp_root_recipe("functional")
    recipe = replace(original, bindings=(p.GPRootBinding(0, columns),))
    actual = p.generate_task(recipe, seed=47, task_id=59)
    wanted = np.asarray(recipe.source.draw.values)[:, list(columns)]
    np.testing.assert_array_equal(actual.inputs, wanted)
    np.testing.assert_allclose(actual.targets, 2 * wanted[:, 0] - wanted[:, 1], atol=1e-14)


@pytest.mark.parametrize("case", ["seed", "task", "device"])
def test_gp_root_call_identity_precedes_replay(case: str, monkeypatch: pytest.MonkeyPatch):
    recipe = gp_root_recipe("functional")

    def forbidden(*args: object, **kwargs: object):
        pytest.fail("invalid call must reject before GP covariance replay")

    monkeypatch.setattr(np.linalg, "cholesky", forbidden)
    with pytest.raises(ValueError, match="identity|device"):
        p.generate_task(
            recipe,
            seed=48 if case == "seed" else 47,
            task_id=60 if case == "task" else 59,
            device="invalid" if case == "device" else "cpu",
        )


@pytest.mark.parametrize("family", ["functional", "native", "vector", "converted"])
def test_gp_root_trace_retains_external_rows_and_target(family: str):
    recipe = gp_root_recipe(family)
    task = p.generate_task(recipe, seed=47, task_id=59)
    trace = p.explain_generation(task, [11, 2, 11])
    assert trace["kind"] == "gp_root_dag"
    assert trace["rows"] == [11, 2, 11]
    nodes = {n["id"]: n for n in trace["nodes"]}
    np.testing.assert_array_equal(nodes["gp-population"]["values"], task.inputs[[11, 2, 11]])
    target = nodes.get("target", nodes.get("targets"))
    assert target is not None
    np.testing.assert_allclose(
        np.asarray(target["values"])[:, 0], task.targets[[11, 2, 11]], atol=1e-14
    )
    seen: set[str] = set()
    for node in trace["nodes"]:
        assert node["id"] not in seen
        assert set(node["parents"]) <= seen
        seen.add(node["id"])


@pytest.mark.parametrize("field", ["task_type", "target_origin", "targets", "seed"])
def test_gp_root_trace_validates_complete_task_custody(field: str):
    from dataclasses import replace

    recipe = gp_root_recipe("native" if field == "target_origin" else "functional")
    task = p.generate_task(recipe, seed=47, task_id=59)
    if field == "task_type":
        forged = replace(task, task_type="binary")
    elif field == "target_origin":
        forged = replace(task, target_origin="adapter-v1")
    elif field == "targets":
        changed = task.targets.clone()
        changed[0] += 1.0
        forged = replace(task, targets=changed)
    else:
        record = json.loads(task.recipe_json)
        record["seed"] = float(record["seed"])
        forged = replace(task, recipe_json=json.dumps(record))
    with pytest.raises(ValueError, match="GP"):
        p.explain_generation(forged, [11, 2, 11])


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("cpu_threads", True),
        ("device", "invalid"),
        ("implementation", "wrong"),
        ("python", 3),
        ("gpu", 1),
        ("extra", "unexpected"),
    ],
)
def test_gp_root_execution_header_rejects_before_replay(
    field: str,
    value: object,
    monkeypatch: pytest.MonkeyPatch,
):
    from dataclasses import replace

    task = p.generate_task(gp_root_recipe("functional"), seed=47, task_id=59)
    record = json.loads(task.recipe_json)
    record["execution"][field] = value
    forged = replace(task, recipe_json=json.dumps(record))

    def forbidden(*args: object, **kwargs: object):
        pytest.fail("malformed execution header began covariance replay")

    monkeypatch.setattr(np.linalg, "cholesky", forbidden)
    with pytest.raises(ValueError, match="GP execution"):
        p.explain_generation(forged, [2])


@pytest.mark.parametrize("transport", ["typed", "json"])
@pytest.mark.parametrize("change", ["seed", "start", "count", "float_seed", "device", "duration"])
def test_gp_root_run_identity_before_replay(
    transport: str,
    change: str,
    monkeypatch: pytest.MonkeyPatch,
):
    from r0.modules import generation as g

    recipe = gp_root_recipe("functional")
    fields: dict[str, Any] = dict(seed=47, count=1, start=59, max_seconds=30, max_bytes=10_000_000)
    mutations = {
        "seed": ("seed", 48),
        "start": ("start", 60),
        "count": ("count", 2),
        "float_seed": ("seed", 47.0),
        "device": ("device", "invalid"),
        "duration": ("max_seconds", -1),
    }
    key, value = mutations[change]
    fields[key] = value

    def forbidden(*args: object, **kwargs: object):
        pytest.fail("invalid run transport began covariance replay")

    monkeypatch.setattr(np.linalg, "cholesky", forbidden)
    if transport == "typed":
        with pytest.raises(ValueError, match="identity|integers|device|seconds"):
            g.RunRequest(recipe, **fields)
    else:
        with pytest.raises(ValueError, match="identity|integers|device|seconds"):
            g.request_from_dict({"recipe": asdict(recipe), **fields})


@pytest.mark.parametrize("change", ["task_type", "target", "execution"])
@pytest.mark.parametrize("calibrated", [False, True])
def test_gp_root_runner_rejects_forged_producer(tmp_path: Path, change: str, calibrated: bool):
    from dataclasses import replace

    from r0.modules import generation as g

    original = gp_root_recipe("functional")
    recipe = (
        p.calibrate_gp_root_dag(
            original,
            calibration_rows=32,
            calibration_seed=101,
            calibration_task_id=103,
        )
        if calibrated
        else original
    )
    task = p.generate_task(recipe, seed=47, task_id=59)
    if change == "task_type":
        forged = replace(task, task_type="binary")
    elif change == "target":
        targets = task.targets.clone()
        targets[0] += 1
        forged = replace(task, targets=targets)
    else:
        record = json.loads(task.recipe_json)
        record["execution"]["cpu_threads"] = True
        forged = replace(task, recipe_json=json.dumps(record))
    result = g.run_generation(
        g.RunRequest(recipe, seed=47, count=1, start=59, max_seconds=30, max_bytes=10_000_000),
        tmp_path,
        produce=lambda _: forged,
    )
    assert result.completed == 0
    assert result.failed == 1


@pytest.mark.parametrize("family", ["functional", "native", "vector", "converted"])
def test_gp_root_runner_roundtrip_and_resume(tmp_path: Path, family: str):
    from r0.modules import generation as g

    recipe = gp_root_recipe(family)
    request = g.request_from_dict(
        {
            "recipe": asdict(recipe),
            "seed": 47,
            "start": 59,
            "count": 1,
            "max_seconds": 30,
            "max_bytes": 10_000_000,
        }
    )
    result = g.run_generation(request, tmp_path)
    assert (result.completed, result.failed) == (1, 0)
    archive = next(tmp_path.rglob("data-*.zip"))
    g.verify_group(archive)
    restored = next(g.load_group(archive))
    p.validate_gp_root_task(restored)
    np.testing.assert_array_equal(restored.inputs, np.asarray(recipe.source.draw.values))
    np.testing.assert_allclose(
        restored.targets,
        2 * restored.inputs[:, 0] - restored.inputs[:, 1],
        atol=1e-14,
    )
    p.explain_generation(restored, [11, 2, 11])
    before = archive.read_bytes()
    resumed = g.run_generation(request, tmp_path)
    assert resumed.failed == 0
    assert archive.read_bytes() == before
