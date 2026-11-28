"""Fresh-query metadata admission includes the prospective retained wrapper."""

from typing import Any

import pytest

from r0.modules import priors as p


@pytest.fixture(scope="module")
def source():
    limits = p.CausalLimits(parameters=16_000_000, working_bytes=4 * 1024**3)
    task = p.LinearRecipe(rows=16, context_rows=8, features=1, signal_features=1, active_features=1)
    f = p.FrozenLinearFunction(((1.0,),))
    base = p.FunctionalCausalRecipe(
        task,
        (
            p.FunctionalCausalNode((), f, p.JointRootBase("gaussian", 1)),
            p.FunctionalCausalNode((0,), f),
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
        source="r0-extension-v1",
        leaf_mean=(0.1,),
        leaf_scale=(1.0,),
    )
    original = p.realize_recursive_population_graph(
        p.RecursivePopulationGraphRequest(
            base, (p.RecursiveNodePopulationPlan(1, forest),), limits=limits
        ),
        seed=31,
        task_id=2**55 + 7,
    )
    return p.calibrate_population_graph(
        original, calibration_rows=32, calibration_seed=41, calibration_task_id=43
    )


def test_next_wrapper_depth_rejects_before_restoration(
    source: p.CalibratedPopulationGraph, monkeypatch: pytest.MonkeyPatch
):
    current: p.CalibratedPopulationGraph | p.CalibratedPopulationQuery = source
    for depth in range(1, 13):
        current = p.realize_calibrated_population_query(
            current, seed=53, task_id=2**55 + 101 + depth, construction_seed=59
        )
    assert isinstance(current, p.CalibratedPopulationQuery)
    assert len(current.lineage()[1]) == 12

    def forbidden(*args: Any, **kwargs: Any):
        pytest.fail("inadmissible prospective query restored a retained stream")

    monkeypatch.setattr(type(source.original.bindings[0].sample.initial), "restore", forbidden)
    with pytest.raises(ValueError, match="traversal|depth"):
        p.realize_calibrated_population_query(
            current, seed=53, task_id=2**55 + 114, construction_seed=59
        )


@pytest.mark.parametrize("resource", ["working_bytes", "parameters"])
def test_raw_prospective_exact_resource_boundary_has_no_replay(
    source: p.CalibratedPopulationGraph, monkeypatch: pytest.MonkeyPatch, resource: str
):
    from dataclasses import asdict, replace

    def forbidden(*args: Any, **kwargs: Any):
        pytest.fail("metadata-only query admission restored a stream")

    monkeypatch.setattr(type(source.original.bindings[0].sample.initial), "restore", forbidden)
    import numpy as np

    for engine in ("PCG64", "PCG64DXSM", "Philox", "SFC64", "MT19937"):
        monkeypatch.setattr(np.random, engine, forbidden)
    raw = asdict(source)
    controls: dict[str, Any] = dict(
        seed=53, task_id=2**55 + 103, engine="MT19937", construction_seed=59
    )
    lower, upper = 1, getattr(source.limits, resource)
    assert p.preflight_calibrated_query(raw, limits=source.limits, **controls) == source.limits
    while lower < upper:
        middle = (lower + upper) // 2
        cap = replace(source.limits, **{resource: middle})
        try:
            p.preflight_calibrated_query(raw, limits=cap, **controls)
        except ValueError:
            lower = middle + 1
        else:
            upper = middle
    cap = replace(source.limits, **{resource: upper})
    assert p.preflight_calibrated_query(raw, limits=cap, **controls) == cap
    with pytest.raises(ValueError, match="limit|metadata"):
        p.preflight_calibrated_query(
            raw, limits=replace(source.limits, **{resource: upper - 1}), **controls
        )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("seed", True),
        ("task_id", 2**63),
        ("construction_seed", -1),
        ("engine", "unknown"),
        ("limits", {}),
        ("task_id", 2**55 + 7),
    ],
)
def test_raw_prospective_controls_precede_restore(
    source: p.CalibratedPopulationGraph, monkeypatch: pytest.MonkeyPatch, field: str, value: Any
):
    from dataclasses import asdict

    def forbidden(*args: Any, **kwargs: Any):
        pytest.fail("invalid prospective query metadata restored a stream")

    monkeypatch.setattr(type(source.original.bindings[0].sample.initial), "restore", forbidden)
    controls: dict[str, Any] = dict(
        seed=31, task_id=2**55 + 103, engine="PCG64", construction_seed=59, limits=source.limits
    )
    controls[field] = value
    with pytest.raises(ValueError, match="query|integer|identity|limits"):
        p.preflight_calibrated_query(asdict(source), **controls)
