"""Population target scaling uses retained independent rows, never query statistics."""

import json
from dataclasses import asdict
from pathlib import Path
from typing import Literal

import numpy as np
import pytest

from r0.modules import priors as p


def _original(
    family: str,
    task_type: Literal["regression", "binary", "multiclass"] = "regression",
    constant: bool = False,
    root_family: str = "linear",
):
    width = 3 if task_type == "multiclass" else 1
    matrix = tuple(tuple(float(i == j) for j in range(width)) for i in range(width))
    controls = p.LinearRecipe(
        rows=16,
        context_rows=7,
        features=1,
        signal_features=1,
        active_features=1,
        task_type=task_type,
        classes={"regression": 0, "binary": 2, "multiclass": 3}[task_type],
        noise_ratio=0.0,
    )
    function = p.FrozenLinearFunction(matrix)
    if root_family == "network":
        function = p.FrozenNetworkFunction(
            width, (p.NetworkBlock((0,), (p.NetworkAffine(matrix, (0.4,) * width),)),)
        )
    elif root_family == "sampled-convolution":
        function = p.sample_feature_conv1d(
            p.FeatureConv1DRequest(width, 1, (p.Conv1DLayerPrior((p.Conv1DGeometry(1, 1),)),)),
            named_rng=np.random.default_rng(31),
            raw_rng=np.random.default_rng(41),
        )
    nodes = (
        p.FunctionalCausalNode((), function, p.JointRootBase("gaussian", width)),
        p.FunctionalCausalNode((0,), p.FrozenLinearFunction(matrix)),
        p.FunctionalCausalNode(
            (1,), p.FrozenLinearFunction(tuple(tuple(3 * v for v in row) for row in matrix))
        ),
    )
    if family == "functional":
        base = p.FunctionalCausalRecipe(
            controls, nodes, ((0, 0),), tuple((2, j) for j in range(width))
        )
    elif family == "native":
        base = p.NativeCausalRecipe(
            controls, tuple(p.NativeCausalNode(n) for n in nodes), ((0, 0),), (2, 0)
        )
    elif family == "converted":
        base = p.ConvertedCausalRecipe(
            controls,
            tuple(p.ConvertedCausalNode(n) for n in nodes),
            (p.ConvertedReadout(0, channel=0),),
            p.ConvertedReadout(2, channel=0),
        )
    else:
        base = p.NativeVectorRecipe(
            controls,
            tuple(p.NativeVectorNode(n, (p.VariableDomain(),)) for n in nodes),
            ((0, 0),),
            (2, 0),
        )
    plan = p.RecursiveForestPlan(
        input_width=width,
        intervals=((-1.0, 1.0),) * width,
        leaf_mean=(0.1,) * width,
        leaf_scale=((0.0 if constant else 1.0),) * width,
        source="r0-extension-v1",
        estimators=1,
        min_depth=1,
        max_depth=1,
        lifecycle="fresh-forest-per-row",
        populations=16,
    )
    original = p.realize_recursive_population_graph(
        p.RecursivePopulationGraphRequest(base, (p.RecursiveNodePopulationPlan(1, plan),)),
        seed=37,
        task_id=2**55 + 9,
    )
    return original


@pytest.mark.parametrize("family", ["functional", "native", "converted", "vector"])
def test_population_target_calibration_uses_independent_population(family: str):
    original = _original(family)
    calibrated = p.calibrate_population_graph(
        original, calibration_rows=32, calibration_seed=71, calibration_task_id=2**55 + 11
    )
    fit = calibrated.calibration_population
    sample = p.generate_task(fit, seed=fit.seed, task_id=fit.task_id)
    before = p.generate_task(original, seed=original.seed, task_id=original.task_id)
    after = p.generate_task(calibrated, seed=original.seed, task_id=original.task_id)
    raw = sample.scores.numpy()
    np.testing.assert_allclose(calibrated.calibration.center, raw.mean(axis=0), atol=1e-14)
    np.testing.assert_allclose(calibrated.calibration.scale, raw.std(axis=0), atol=1e-14)
    np.testing.assert_allclose(
        after.scores, (before.scores.numpy() - raw.mean(0)) / raw.std(0), atol=1e-14
    )
    np.testing.assert_array_equal(before.inputs, after.inputs)
    np.testing.assert_array_equal(before.context_indices, after.context_indices)
    np.testing.assert_allclose(after.targets, after.scores[:, 0], atol=1e-14)
    assert fit.task.rows == 32
    assert fit.seed != original.seed

    restored = p.recipe_from_dict(json.loads(json.dumps(asdict(calibrated))))
    replay = p.generate_task(restored, seed=original.seed, task_id=original.task_id)
    np.testing.assert_array_equal(replay.targets, after.targets)
    assert replay.recipe_json == after.recipe_json


@pytest.mark.parametrize("family", ["functional", "native", "converted", "vector"])
def test_calibrated_population_original_row_trace(family: str):
    original = _original(family)
    calibrated = p.calibrate_population_graph(
        original, calibration_rows=32, calibration_seed=71, calibration_task_id=2**55 + 11
    )
    task = p.generate_task(calibrated, seed=original.seed, task_id=original.task_id)
    trace = p.explain_generation(task, [15, 2, 15])
    nodes = {node["id"]: node for node in trace["nodes"]}
    assert trace["rows"] == [15, 2, 15]
    np.testing.assert_array_equal(nodes["signal"]["values"], task.scores[[15, 2, 15]])
    assert nodes["calibration-population"]["shape"][0] == 32
    assert "independent" in nodes["calibration-population"]["parameters"]["scope"]
    seen: set[str] = set()
    for node in trace["nodes"]:
        assert node["id"] not in seen
        assert set(node["parents"]) <= seen
        seen.add(node["id"])


@pytest.fixture(scope="module")
def calibrated_fixture() -> p.CalibratedPopulationGraph:
    return p.calibrate_population_graph(
        _original("functional"),
        calibration_rows=32,
        calibration_seed=71,
        calibration_task_id=2**55 + 11,
    )


@pytest.mark.parametrize(
    ("side", "path", "value"),
    [
        ("original", ("seed",), True),
        ("calibration_population", ("task_id",), 19.0),
        ("calibration_population", ("request", "base", "task", "rows"), True),
        ("calibration_population", ("bindings", 0, "node"), True),
        ("calibration_population", ("bindings", 0, "inputs"), ((0.0,),)),
        ("calibration_population", ("bindings", 0, "sample", "raw", 0, "sequence"), True),
        ("calibration_population", ("bindings", 0, "sample", "final", "digest"), "0" * 64),
    ],
)
def test_calibration_malformed_population_rejects_before_restore(
    calibrated_fixture: p.CalibratedPopulationGraph,
    side: str,
    path: tuple[str | int, ...],
    value: object,
):
    from unittest.mock import patch

    raw = asdict(calibrated_fixture)
    destination = raw[side]
    for key in path[:-1]:
        destination = destination[key]
    destination[path[-1]] = value
    cls = type(calibrated_fixture.original.bindings[0].sample.initial)
    with (
        patch.object(cls, "restore", side_effect=AssertionError("restored before preflight")),
        pytest.raises(ValueError, match="."),
    ):
        p.recipe_from_dict(raw)


@pytest.mark.parametrize("field", ["task_type", "source_task_id", "episode_id", "scores"])
def test_calibration_trace_rejects_whole_task_semantics(
    calibrated_fixture: p.CalibratedPopulationGraph,
    field: str,
):
    from dataclasses import replace
    from typing import Any

    task = p.generate_task(calibrated_fixture, seed=37, task_id=2**55 + 9)
    changes: dict[str, Any] = {field: task.scores.float() if field == "scores" else "forged"}
    with pytest.raises(ValueError, match="semantic"):
        p.explain_generation(replace(task, **changes), [0])


def test_calibration_aggregate_budget_precedes_restore(
    calibrated_fixture: p.CalibratedPopulationGraph,
):
    from dataclasses import replace
    from unittest.mock import patch

    cls = type(calibrated_fixture.original.bindings[0].sample.initial)
    with patch.object(cls, "restore", side_effect=AssertionError("restored before budget")):
        with pytest.raises(ValueError, match="limit|budget|metadata|caps"):
            p.recipe_from_dict(
                asdict(calibrated_fixture),
                limits=replace(calibrated_fixture.limits, working_bytes=1024),
            )
        with pytest.raises(ValueError, match="limit|budget|metadata|caps"):
            p.calibrate_population_graph(
                calibrated_fixture.original,
                calibration_rows=32,
                calibration_seed=71,
                calibration_task_id=2**55 + 11,
                limits=replace(calibrated_fixture.limits, working_bytes=1024),
            )


@pytest.mark.parametrize("task_type", ["binary", "multiclass"])
def test_functional_calibration_classification_and_zip(
    task_type: Literal["binary", "multiclass"],
    tmp_path: Path,
):
    from r0.modules import generation as g

    original = _original("functional", task_type)
    recipe = p.calibrate_population_graph(
        original, calibration_rows=32, calibration_seed=101, calibration_task_id=103
    )
    task = p.generate_task(recipe, seed=original.seed, task_id=original.task_id)
    scores = task.scores.numpy()
    logits = original.task.separation * scores
    if task_type == "binary":
        positive = 1 / (1 + np.exp(-logits[:, 0]))
        expected = np.column_stack((1 - positive, positive))
    else:
        exp = np.exp(logits - logits.max(axis=1, keepdims=True))
        expected = exp / exp.sum(axis=1, keepdims=True)
    assert task.probabilities is not None
    np.testing.assert_allclose(task.probabilities.numpy(), expected, atol=1e-14)
    graph = p.explain_generation(task, [15, 0, 15])
    assert graph["target_max_error"] == 0
    path = tmp_path / "calibrated.zip"
    g.save_group(path, [(original.task_id, task)], max_bytes=64 * 1024**2)
    restored = next(g.load_group(path))
    assert restored.recipe_json == task.recipe_json
    assert restored.mechanism_json == task.mechanism_json
    assert p.explain_generation(restored, [15, 0, 15])["target_max_error"] == 0


def test_exact_constant_calibration_retains_identity():
    original = _original("functional", constant=True)
    recipe = p.calibrate_population_graph(
        original, calibration_rows=37, calibration_seed=101, calibration_task_id=103
    )
    assert recipe.calibration.constant_channels == (0,)
    assert recipe.calibration.center == (0.0,)
    assert recipe.calibration.scale == (1.0,)
    before = p.generate_task(original, seed=original.seed, task_id=original.task_id)
    after = p.generate_task(recipe, seed=original.seed, task_id=original.task_id)
    np.testing.assert_array_equal(before.scores, after.scores)


def test_calibration_preserves_original_observation_noise_draws():
    from dataclasses import replace

    base = _original("functional")
    request = replace(
        base.request, base=replace(base.request.base, task=replace(base.task, noise_ratio=0.35))
    )
    original = p.realize_recursive_population_graph(request, seed=37, task_id=2**55 + 9)
    recipe = p.calibrate_population_graph(
        original, calibration_rows=32, calibration_seed=101, calibration_task_id=103
    )
    before = p.generate_task(original, seed=original.seed, task_id=original.task_id)
    after = p.generate_task(recipe, seed=original.seed, task_id=original.task_id)
    np.testing.assert_allclose(
        (after.targets - after.scores[:, 0]).numpy(),
        (before.targets - before.scores[:, 0]).numpy(),
        atol=1e-14,
    )
    assert p.explain_generation(after, [15, 0, 15])["target_max_error"] == 0


@pytest.mark.parametrize("family", ["functional", "native", "converted", "vector"])
@pytest.mark.parametrize("root_family", ["network", "sampled-convolution"])
def test_calibration_broad_frozen_mechanisms_preserve_original(family: str, root_family: str):
    original = _original(family, root_family=root_family)
    snapshot = asdict(original)
    recipe = p.calibrate_population_graph(
        original, calibration_rows=32, calibration_seed=107, calibration_task_id=109
    )
    assert asdict(recipe.original) == snapshot
    raw = asdict(recipe)
    restored = p.recipe_from_dict(raw)
    assert asdict(restored) == raw
    task = p.generate_task(restored, seed=original.seed, task_id=original.task_id)
    assert p.explain_generation(task, [15, 0, 15])["target_max_error"] == 0


@pytest.mark.parametrize("mutation", ["prefix", "integer-center", "identity", "same-identity"])
def test_calibration_bindings_reject_before_restore(
    calibrated_fixture: p.CalibratedPopulationGraph,
    mutation: str,
):
    from unittest.mock import patch

    raw = asdict(calibrated_fixture)
    if mutation == "prefix":
        raw["calibration"]["prefix_fingerprint"] = "0" * 64
    elif mutation == "integer-center":
        raw["calibration"]["center"] = (0,)
    elif mutation == "identity":
        raw["calibration_population"]["seed"] += 1
    else:
        raw["calibration_population"]["seed"] = raw["original"]["seed"]
        raw["calibration_population"]["task_id"] = raw["original"]["task_id"]
    cls = type(calibrated_fixture.original.bindings[0].sample.initial)
    with (
        patch.object(cls, "restore", side_effect=AssertionError("binding reached restore")),
        pytest.raises(ValueError, match="calibration"),
    ):
        p.recipe_from_dict(raw)


def test_direct_categorical_target_retains_its_unscaled_semantics():
    from dataclasses import replace

    original = _original("native")
    base = original.request.base
    assert isinstance(base, p.NativeCausalRecipe)
    nodes = (
        *base.nodes[:-1],
        p.NativeCausalNode(
            replace(base.nodes[-1].equation, function=p.FrozenLinearFunction(((1.0,), (-1.0,)))),
            p.VariableDomain("binary", ("no", "yes"), "softmax"),
        ),
    )
    classified = p.realize_recursive_population_graph(
        replace(
            original.request,
            base=replace(base, task=replace(base.task, task_type="binary", classes=2), nodes=nodes),
        ),
        seed=37,
        task_id=2**55 + 9,
    )
    before = p.generate_task(classified, seed=classified.seed, task_id=classified.task_id)
    with pytest.raises(ValueError, match="category semantics"):
        p.calibrate_population_graph(
            classified, calibration_rows=32, calibration_seed=101, calibration_task_id=103
        )
    after = p.generate_task(classified, seed=classified.seed, task_id=classified.task_id)
    assert before.recipe_json == after.recipe_json
    np.testing.assert_array_equal(before.targets, after.targets)
    assert before.classes == ("no", "yes")


@pytest.mark.parametrize("mutation", ["valid", "task-type", "seed-type"])
def test_run_preserves_complete_calibrated_task(
    calibrated_fixture: p.CalibratedPopulationGraph, tmp_path: Path, mutation: str
):
    from dataclasses import replace

    from r0.modules import generation as g

    recipe = calibrated_fixture
    task = p.generate_task(recipe, seed=recipe.original.seed, task_id=recipe.original.task_id)
    if mutation == "task-type":
        task = replace(task, task_type="binary")
    elif mutation == "seed-type":
        record = json.loads(task.recipe_json)
        record["seed"] = float(record["seed"])
        task = replace(task, recipe_json=json.dumps(record))
    result = g.run_generation(
        g.RunRequest(
            recipe,
            recipe.original.seed,
            1,
            120,
            64 * 1024**2,
            start=recipe.original.task_id,
        ),
        tmp_path,
        produce=lambda _: task,
    )
    if mutation == "valid":
        assert result.completed == 1
        assert result.failed == 0
        archive = next(tmp_path.rglob("data-*.zip"))
        restored = next(g.load_group(archive))
        assert restored.recipe_json == task.recipe_json
        assert p.explain_generation(restored, [15, 0, 15])["target_max_error"] == 0
    else:
        assert result.completed == 0
        assert result.failed == 1


@pytest.mark.parametrize("mutation", ["choice", "chain", "snapshot"])
def test_nested_convolution_custody_precedes_any_restore(mutation: str):
    from unittest.mock import patch

    original = _original("functional", root_family="sampled-convolution")
    calibrated = p.calibrate_population_graph(
        original, calibration_rows=32, calibration_seed=71, calibration_task_id=73
    )
    raw = json.loads(json.dumps(asdict(calibrated)))
    convolution = raw["calibration_population"]["request"]["base"]["nodes"][0]["function"]
    if mutation == "choice":
        convolution["choices"][0][0] = 99
    elif mutation == "chain":
        convolution["raw"][0]["state_before"] = "0" * 64
    else:
        convolution["initial"][0]["numpy_version"] = False
    cls = type(original.bindings[0].sample.initial)
    with (
        patch.object(cls, "restore", side_effect=AssertionError("nested custody reached restore")),
        pytest.raises(ValueError, match="calibration|snapshot|convolution"),
    ):
        p.recipe_from_dict(raw)
