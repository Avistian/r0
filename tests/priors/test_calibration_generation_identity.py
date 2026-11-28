"""Known frozen identity/transport failures precede either independent population replay."""

from dataclasses import asdict, replace
from typing import Any

import pytest

from r0.modules import generation as g
from r0.modules import priors as p


@pytest.fixture(scope="module")
def calibrated():
    task = p.LinearRecipe(rows=16, context_rows=7, features=1, signal_features=1, active_features=1)
    nodes = (
        p.FunctionalCausalNode(
            (), p.FrozenLinearFunction(((1.0,),)), p.JointRootBase("gaussian", 1)
        ),
        p.FunctionalCausalNode((0,), p.FrozenLinearFunction(((1.0,),))),
        p.FunctionalCausalNode((1,), p.FrozenLinearFunction(((-2.3,),))),
    )
    base = p.FunctionalCausalRecipe(task, nodes, ((0, 0),), ((2, 0),))
    forest = p.RecursiveForestPlan(
        input_width=1,
        intervals=((-1.0, 1.0),),
        leaf_mean=(0.1,),
        leaf_scale=(1.0,),
        source="r0-extension-v1",
        estimators=1,
        min_depth=1,
        max_depth=1,
        lifecycle="fresh-forest-per-row",
        populations=16,
    )
    original = p.realize_recursive_population_graph(
        p.RecursivePopulationGraphRequest(base, (p.RecursiveNodePopulationPlan(1, forest),)),
        seed=91,
        task_id=2**55 + 19,
    )
    return p.calibrate_population_graph(
        original, calibration_rows=32, calibration_seed=101, calibration_task_id=2**55 + 21
    )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("seed", 92),
        ("seed", True),
        ("seed", 91.0),
        ("seed", -1),
        ("seed", 2**63),
        ("task_id", 3),
        ("task_id", False),
        ("task_id", float(2**55 + 19)),
        ("task_id", -1),
        ("task_id", 2**63),
        ("device", "cuda"),
        ("device", "other"),
        ("device", False),
    ],
)
def test_typed_generation_identity_before_restore(
    calibrated: p.CalibratedPopulationGraph,
    monkeypatch: pytest.MonkeyPatch,
    field: str,
    value: Any,
):
    def forbidden(*args: Any, **kwargs: Any):
        pytest.fail("known generation identity/device must reject before restore")

    monkeypatch.setattr(type(calibrated.original.bindings[0].sample.initial), "restore", forbidden)
    parameters: dict[str, Any] = dict(seed=91, task_id=2**55 + 19, device="cpu")
    parameters[field] = value
    with pytest.raises(ValueError, match="identity|CPU|integer"):
        p.generate_task(calibrated, **parameters)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("seed", 92),
        ("seed", True),
        ("seed", 91.0),
        ("seed", -1),
        ("seed", 2**63),
        ("start", 3),
        ("start", False),
        ("start", float(2**55 + 19)),
        ("start", -1),
        ("start", 2**63),
        ("device", "cuda"),
        ("device", "other"),
        ("device", False),
        ("count", 2),
        ("count", True),
        ("count", 0),
        ("max_seconds", float("nan")),
        ("max_bytes", 1),
        ("group_size", 0),
        ("sampling", {"family": "linear"}),
        ("sampling", {}),
        ("construction", {}),
    ],
)
def test_raw_generation_request_rejects_before_source_decode(
    calibrated: p.CalibratedPopulationGraph,
    monkeypatch: pytest.MonkeyPatch,
    field: str,
    value: Any,
):
    raw: dict[str, Any] = dict(
        recipe=asdict(calibrated),
        seed=91,
        start=2**55 + 19,
        count=1,
        max_seconds=60,
        max_bytes=256 * 1024**2,
        device="cpu",
    )
    raw[field] = value

    def forbidden(*args: Any, **kwargs: Any):
        pytest.fail("known transport metadata must reject before restore")

    monkeypatch.setattr(type(calibrated.original.bindings[0].sample.initial), "restore", forbidden)
    with pytest.raises(
        ValueError,
        match="identity|CPU|integer|interval|count|storage|seconds|sampling|construction|group",
    ):
        g.request_from_dict(raw)


def test_valid_typed_and_raw_generation_preserve_exact_task(
    calibrated: p.CalibratedPopulationGraph,
):
    request = g.RunRequest(calibrated, 91, 1, 60, 256 * 1024**2, start=2**55 + 19)
    decoded = g.request_from_dict(asdict(request))
    before = p.generate_task(calibrated, seed=91, task_id=2**55 + 19)
    after = p.generate_task(decoded.recipe, seed=91, task_id=2**55 + 19)
    assert before.recipe_json == after.recipe_json
    assert before.mechanism_json == after.mechanism_json
    assert before.targets.equal(after.targets)


@pytest.mark.parametrize("raw", [False, True])
@pytest.mark.parametrize("origin", [None, False, {}])
def test_malformed_retained_origin_rejects_before_access_or_restore(
    calibrated: p.CalibratedPopulationGraph,
    monkeypatch: pytest.MonkeyPatch,
    raw: bool,
    origin: Any,
):
    def forbidden(*args: Any, **kwargs: Any):
        pytest.fail("malformed retained origin must reject before restore")

    monkeypatch.setattr(type(calibrated.original.bindings[0].sample.initial), "restore", forbidden)
    source = replace(calibrated, original=origin)
    if raw:
        request = dict(
            recipe=asdict(source),
            seed=91,
            start=2**55 + 19,
            count=1,
            max_seconds=60,
            max_bytes=256 * 1024**2,
        )
        with pytest.raises(ValueError, match="identity"):
            g.request_from_dict(request)
    else:
        with pytest.raises(ValueError, match="identity"):
            p.generate_task(source, seed=91, task_id=2**55 + 19)


@pytest.mark.parametrize(("field", "value"), [("seed", 92), ("count", 2), ("device", "cuda")])
def test_typed_run_request_has_same_fixed_population_guard(
    calibrated: p.CalibratedPopulationGraph,
    monkeypatch: pytest.MonkeyPatch,
    field: str,
    value: Any,
):
    def forbidden(*args: Any, **kwargs: Any):
        pytest.fail("typed frozen request must reject known mismatches before replay")

    monkeypatch.setattr(type(calibrated.original.bindings[0].sample.initial), "restore", forbidden)
    controls: dict[str, Any] = dict(
        seed=91, start=2**55 + 19, count=1, max_seconds=60, max_bytes=256 * 1024**2
    )
    controls[field] = value
    with pytest.raises(ValueError, match="identity|CPU|count"):
        g.RunRequest(calibrated, **controls)


@pytest.mark.parametrize("path", ["request", "base", "task", "class_offsets"])
@pytest.mark.parametrize("value", [None, False, {}])
def test_malformed_task_preview_rejects_without_replay(
    calibrated: p.CalibratedPopulationGraph,
    monkeypatch: pytest.MonkeyPatch,
    path: str,
    value: Any,
):
    def forbidden(*args: Any, **kwargs: Any):
        pytest.fail("malformed task metadata reached replay")

    monkeypatch.setattr(type(calibrated.original.bindings[0].sample.initial), "restore", forbidden)
    source = asdict(calibrated)
    node = source["original"]
    for key in ("request", "base", "task", "class_offsets"):
        if key == path:
            node[key] = value
            break
        node = node[key]
    with pytest.raises(ValueError, match="metadata|class_offsets"):
        g.request_from_dict(
            dict(
                recipe=source,
                seed=91,
                start=2**55 + 19,
                count=1,
                max_seconds=60,
                max_bytes=256 * 1024**2,
            )
        )
