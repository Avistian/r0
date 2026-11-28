"""Known retained pair identities reject before any snapshot restoration."""

from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest

from r0.modules import priors as p


def _changed():
    limits = p.CausalLimits(parameters=8_000_000, working_bytes=2 * 1024**3)
    task = p.LinearRecipe(rows=16, context_rows=7, features=1, signal_features=1, active_features=1)
    base = p.FunctionalCausalRecipe(
        task,
        (
            p.FunctionalCausalNode(
                (), p.FrozenLinearFunction(((1.0,),)), p.JointRootBase("gaussian", 1)
            ),
            p.FunctionalCausalNode((0,), p.FrozenLinearFunction(((1.0,),))),
        ),
        ((0, 0),),
        ((1, 0),),
        limits=limits,
    )
    forest = p.RecursiveForestPlan(
        input_width=1,
        intervals=((-1.0, 1.0),),
        estimators=1,
        min_depth=1,
        max_depth=1,
        lifecycle="fresh-forest-per-row",
        populations=16,
    )
    original = p.realize_recursive_population_graph(
        p.RecursivePopulationGraphRequest(
            base, (p.RecursiveNodePopulationPlan(1, forest),), limits=limits
        ),
        seed=37,
        task_id=2**55 + 9,
    )
    calibrated = p.calibrate_population_graph(
        original, calibration_rows=32, calibration_seed=71, calibration_task_id=73
    )
    return p.intervene_calibrated_population(
        calibrated, p.CausalInterventionRecipe(base, (p.HardIntervention(1, (0,), (2.5,)),))
    )


@pytest.fixture(scope="module")
def changed() -> p.CalibratedPopulationIntervention:
    return _changed()


@pytest.mark.parametrize("mutation", ["seed", "task-id", "float-seed", "bool-task-id", "device"])
def test_pair_generation_identity_before_restore(
    changed: p.CalibratedPopulationIntervention, mutation: str
):
    seed, task_id, device = 37, 2**55 + 9, "cpu"
    kwargs: dict[str, Any] = {"seed": seed, "task_id": task_id, "device": device}
    if mutation == "seed":
        kwargs["seed"] = 38
    elif mutation == "task-id":
        kwargs["task_id"] = task_id + 1
    elif mutation == "float-seed":
        kwargs["seed"] = 37.0
    elif mutation == "bool-task-id":
        kwargs["task_id"] = True
    else:
        kwargs["device"] = "cuda"
    cls = type(changed.base.original.bindings[0].sample.initial)
    with (
        patch.object(cls, "restore", side_effect=AssertionError("known identity restored RNG")),
        pytest.raises(ValueError, match="identity|identities|integer|CPU"),
    ):
        p.generate_task(changed, **kwargs)


def test_valid_pair_identity_replays_after_guard(changed: p.CalibratedPopulationIntervention):
    task = p.generate_task(changed, seed=37, task_id=2**55 + 9)
    trace = p.explain_generation(task, [15, 0, 15], limits=changed.limits)
    assert trace["target_max_error"] == 0


@pytest.mark.parametrize(
    "mutation", ["seed", "start", "device", "count", "sampling", "construction"]
)
def test_serialized_pair_request_guard_precedes_restore(
    changed: p.CalibratedPopulationIntervention, mutation: str
):
    from dataclasses import asdict

    from r0.modules import generation as g

    raw: dict[str, Any] = {
        "recipe": asdict(changed),
        "seed": 37,
        "start": 2**55 + 9,
        "count": 1,
        "max_seconds": 120,
        "max_bytes": 64 * 1024**2,
        "causal_limits": asdict(changed.limits),
    }
    if mutation == "seed":
        raw["seed"] = 38
    elif mutation == "start":
        raw["start"] = 0
    elif mutation == "device":
        raw["device"] = "cuda"
    elif mutation == "count":
        raw["count"] = 2
    else:
        raw[mutation] = {}
    cls = type(changed.base.original.bindings[0].sample.initial)
    with (
        patch.object(cls, "restore", side_effect=AssertionError("request restored RNG")),
        pytest.raises(ValueError, match="calibrat|identity|count|sampling|construction|CPU"),
    ):
        g.request_from_dict(raw)


@pytest.mark.parametrize("mutation", ["seed", "start", "device", "count"])
def test_typed_pair_request_identity_contract(
    changed: p.CalibratedPopulationIntervention, mutation: str
):
    from r0.modules import generation as g

    raw: dict[str, Any] = {
        "recipe": changed,
        "seed": 37,
        "start": 2**55 + 9,
        "count": 1,
        "max_seconds": 120,
        "max_bytes": 64 * 1024**2,
        "causal_limits": changed.limits,
    }
    raw[mutation] = {"seed": 38, "start": 0, "device": "cuda", "count": 2}[mutation]
    with pytest.raises(ValueError, match="calibrat|identity|count|CPU"):
        g.RunRequest(**raw)


def test_valid_serialized_pair_request_persists(
    changed: p.CalibratedPopulationIntervention, tmp_path: Path
):
    from dataclasses import asdict

    from r0.modules import generation as g

    request = g.RunRequest(
        changed, 37, 1, 120, 64 * 1024**2, start=2**55 + 9, causal_limits=changed.limits
    )
    decoded = g.request_from_dict(asdict(request))
    result = g.run_generation(decoded, tmp_path)
    assert result.completed == 1
    assert result.failed == 0
    task = next(g.load_group(next(tmp_path.rglob("data-*.zip"))))
    assert p.explain_generation(task, [15, 0, 15], limits=changed.limits)["target_max_error"] == 0
