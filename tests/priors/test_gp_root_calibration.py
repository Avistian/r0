"""Independent retained GP-root calibration, replay and transport."""

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from gp_root_fixture import gp_root_recipe

from r0.modules import priors as p


@pytest.mark.parametrize("family", ["functional", "native", "vector", "converted"])
def test_gp_root_independent_target_calibration(family: str):
    recipe = gp_root_recipe(family)
    calibrated = p.calibrate_gp_root_dag(
        recipe,
        calibration_rows=32,
        calibration_seed=101,
        calibration_task_id=103,
    )
    actual = p.generate_calibrated_gp_root(calibrated, seed=47, task_id=59)
    fit_values = np.asarray(calibrated.fit.source.draw.values)
    independent = 2 * fit_values[:, 0] - fit_values[:, 1]
    query = np.asarray(recipe.source.draw.values)
    expected = (2 * query[:, 0] - query[:, 1] - independent.mean()) / independent.std()
    np.testing.assert_allclose(actual.targets, expected, atol=1e-12)
    np.testing.assert_array_equal(actual.inputs, query)
    assert calibrated.fit.source.population_id != recipe.source.population_id
    assert json.loads(actual.mechanism_json)["query_fit"] is False


@pytest.mark.parametrize("case", ["same", "float", "rows", "budget"])
def test_gp_root_calibration_controls_before_draws(case: str, monkeypatch: pytest.MonkeyPatch):
    recipe = gp_root_recipe("functional")

    def forbidden(*args: object, **kwargs: object):
        pytest.fail("invalid calibration controls began covariance replay")

    monkeypatch.setattr(np.linalg, "cholesky", forbidden)
    seed: Any = 47 if case == "same" else 101.0 if case == "float" else 101
    with pytest.raises(ValueError, match="GP|limit|budget|calibration"):
        p.calibrate_gp_root_dag(
            recipe,
            calibration_rows=16 if case == "rows" else 32,
            calibration_seed=seed,
            calibration_task_id=59 if case == "same" else 103,
            limits=p.CausalLimits(working_bytes=1024) if case == "budget" else None,
        )


@pytest.mark.parametrize("mode", ["constant", "binary"])
def test_gp_root_calibration_preserves_adapter_and_constant_semantics(mode: str):
    from dataclasses import replace

    original = gp_root_recipe("functional")
    assert isinstance(original.base, p.FunctionalCausalRecipe)
    if mode == "constant":
        base = replace(
            original.base,
            nodes=(
                original.base.nodes[0],
                p.FunctionalCausalNode(
                    (0,),
                    p.FrozenLinearFunction(((0.0, 0.0),)),
                ),
            ),
        )
    else:
        base = replace(original.base, task=replace(original.task, task_type="binary", classes=2))
    recipe = replace(original, base=base)
    calibrated = p.calibrate_gp_root_dag(
        recipe,
        calibration_rows=32,
        calibration_seed=101,
        calibration_task_id=103,
    )
    actual = p.generate_calibrated_gp_root(calibrated, seed=47, task_id=59)
    if mode == "constant":
        assert calibrated.calibration.center == (0.0,)
        assert calibrated.calibration.scale == (1.0,)
        assert calibrated.calibration.constant_channels == (0,)
        np.testing.assert_array_equal(actual.targets, np.zeros(16))
    else:
        expected = 1 / (1 + np.exp(-recipe.task.separation * actual.scores[:, 0].numpy()))
        assert actual.probabilities is not None
        np.testing.assert_allclose(actual.probabilities[:, 1], expected, atol=1e-14)
        assert actual.classes == ("class-0", "class-1")
        trace = p.explain_generation(actual, [11, 2, 11])
        nodes = {node["id"]: node for node in trace["nodes"]}
        np.testing.assert_allclose(
            nodes["probabilities"]["values"], actual.probabilities[[11, 2, 11]]
        )


@pytest.mark.parametrize("family", ["functional", "native", "vector", "converted"])
def test_calibrated_gp_root_recipe_roundtrip(family: str):
    recipe = p.calibrate_gp_root_dag(
        gp_root_recipe(family),
        calibration_rows=32,
        calibration_seed=101,
        calibration_task_id=103,
    )
    restored = p.recipe_from_dict(json.loads(json.dumps(asdict(recipe))))
    assert asdict(restored) == asdict(recipe)
    actual = p.generate_task(restored, seed=47, task_id=59)
    expected = p.generate_calibrated_gp_root(recipe, seed=47, task_id=59)
    np.testing.assert_array_equal(actual.targets, expected.targets)
    assert actual.recipe_json == expected.recipe_json


@pytest.mark.parametrize(
    "change",
    [
        "identity",
        "rows",
        "graph",
        "law",
        "binding",
        "version",
        "fit_version",
        "prefix",
        "calibration",
        "budget",
    ],
)
def test_calibrated_gp_metadata_rejects_before_both_populations(
    change: str,
    monkeypatch: pytest.MonkeyPatch,
):
    recipe = p.calibrate_gp_root_dag(
        gp_root_recipe("functional"),
        calibration_rows=32,
        calibration_seed=101,
        calibration_task_id=103,
    )
    raw = json.loads(json.dumps(asdict(recipe)))
    cap = p.CausalLimits()
    if change == "identity":
        raw["fit"]["source"]["seed"] = 47
        raw["fit"]["source"]["task_id"] = 59
    elif change == "rows":
        raw["fit"]["base"]["task"]["rows"] = 16
    elif change == "graph":
        raw["fit"]["base"]["nodes"][1]["function"]["matrix"][0][0] = 3
    elif change == "law":
        raw["fit"]["source"]["request"]["location_family"] = "gaussian-r0"
    elif change == "binding":
        raw["fit"]["bindings"][0]["columns"] = [1, 0]
    elif change == "version":
        raw["version"] = "wrong"
    elif change == "fit_version":
        raw["fit"]["version"] = "wrong"
    elif change == "prefix":
        raw["calibration"]["prefix_fingerprint"] = "0" * 64
    elif change == "calibration":
        raw["calibration"]["scale"] = [-1.0]
    else:
        cap = p.CausalLimits(working_bytes=1024)

    def forbidden(*args: object, **kwargs: object):
        pytest.fail("invalid calibration metadata began covariance replay")

    monkeypatch.setattr(np.linalg, "cholesky", forbidden)
    with pytest.raises(ValueError, match="GP|calibration|limit|sizes|version"):
        p.recipe_from_dict(raw, limits=cap)


@pytest.mark.parametrize("family", ["functional", "native", "vector", "converted"])
def test_calibrated_gp_trace_and_run_roundtrip(tmp_path: Path, family: str):
    from r0.modules import generation as g

    recipe = p.calibrate_gp_root_dag(
        gp_root_recipe(family),
        calibration_rows=32,
        calibration_seed=101,
        calibration_task_id=103,
    )
    task = p.generate_task(recipe, seed=47, task_id=59)
    trace = p.explain_generation(task, [11, 2, 11])
    assert trace["kind"] == "gp_root_calibration"
    nodes = {node["id"]: node for node in trace["nodes"]}
    np.testing.assert_allclose(
        np.asarray(nodes["target"]["values"])[:, 0], task.targets[[11, 2, 11]]
    )
    seen: set[str] = set()
    for node in trace["nodes"]:
        assert node["id"] not in seen
        assert set(node["parents"]) <= seen
        seen.add(node["id"])
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
    assert restored.recipe_json == task.recipe_json
    assert p.explain_generation(restored, [11, 2, 11]) == trace


@pytest.mark.parametrize("field", ["seed", "float_seed", "execution", "source_id", "episode_id"])
def test_calibrated_gp_trace_transport_precedes_replay(field: str, monkeypatch: pytest.MonkeyPatch):
    from dataclasses import replace

    recipe = p.calibrate_gp_root_dag(
        gp_root_recipe("functional"),
        calibration_rows=32,
        calibration_seed=101,
        calibration_task_id=103,
    )
    task = p.generate_task(recipe, seed=47, task_id=59)
    record = json.loads(task.recipe_json)
    if field == "source_id":
        forged = replace(task, source_task_id="0" * 64)
    elif field == "episode_id":
        forged = replace(task, episode_id="0" * 64)
    else:
        if field == "execution":
            record["execution"]["cpu_threads"] = True
        else:
            record["seed"] = 48 if field == "seed" else 47.0
        forged = replace(task, recipe_json=json.dumps(record))

    def forbidden(*args: object, **kwargs: object):
        pytest.fail("invalid calibrated transport began covariance replay")

    monkeypatch.setattr(np.linalg, "cholesky", forbidden)
    with pytest.raises(ValueError, match="GP"):
        p.explain_generation(forged, [2])


@pytest.mark.parametrize("transport", ["typed", "json"])
@pytest.mark.parametrize("change", ["seed", "count", "device", "duration"])
def test_calibrated_gp_run_transport_precedes_replay(
    transport: str,
    change: str,
    monkeypatch: pytest.MonkeyPatch,
):
    from r0.modules import generation as g

    recipe = p.calibrate_gp_root_dag(
        gp_root_recipe("functional"),
        calibration_rows=32,
        calibration_seed=101,
        calibration_task_id=103,
    )
    fields: dict[str, Any] = dict(seed=47, count=1, start=59, max_seconds=30, max_bytes=10_000_000)
    key, value = {
        "seed": ("seed", 48),
        "count": ("count", 2),
        "device": ("device", "cuda"),
        "duration": ("max_seconds", -1),
    }[change]
    fields[key] = value

    def forbidden(*args: object, **kwargs: object):
        pytest.fail("invalid calibrated run began covariance replay")

    monkeypatch.setattr(np.linalg, "cholesky", forbidden)
    if transport == "typed":
        with pytest.raises(ValueError, match="GP|seconds"):
            g.RunRequest(recipe, **fields)
    else:
        with pytest.raises(ValueError, match="GP|seconds"):
            g.request_from_dict({"recipe": asdict(recipe), **fields})


@pytest.mark.parametrize("field", ["sampling", "construction"])
def test_calibrated_gp_unused_run_controls_reject_before_parsing(field: str):
    from r0.modules import generation as g

    recipe = p.calibrate_gp_root_dag(
        gp_root_recipe("functional"),
        calibration_rows=32,
        calibration_seed=101,
        calibration_task_id=103,
    )
    with pytest.raises(ValueError, match="GP"):
        g.request_from_dict(
            {
                "recipe": asdict(recipe),
                "seed": 47,
                "count": 1,
                "start": 59,
                "max_seconds": 30,
                "max_bytes": 10_000_000,
                field: {"unused": "not a constructor"},
            }
        )
