"""Complete future fit custody is admitted before original or fitting GP work."""

from dataclasses import asdict
from typing import Any, cast
from unittest.mock import patch

import numpy as np
import pytest

from r0.modules import priors as p


def make_original(
    limits: p.CausalLimits,
    *,
    family: str = "functional",
    rows: int = 32,
    target: str = "linear",
    location: str = "uniform-r0",
) -> p.GPRootDAGRecipe:
    dimension = 1 if location == "grid-r0" else 2
    locations = (
        tuple((float(i), float(i % 3)) for i in range(rows)) if location == "fixed-r0" else ()
    )
    request = p.GPColumnRequest(
        rows=rows,
        target_family=target,
        location_family=location,
        dimension=dimension,
        locations=locations,
        kernels=(p.GPColumnKernel(length_scale=0.03),),
        limits=limits,
    )
    source = p.realize_gp_columns(request, seed=71, task_id=2**55 + 17)
    task = p.LinearRecipe(
        rows=rows, context_rows=rows // 2, features=2, signal_features=2, active_features=2
    )
    root = p.FunctionalCausalNode(
        (), p.FrozenLinearFunction(((1.0, 0.0), (0.0, 1.0))), p.JointRootBase("gaussian", 2)
    )
    child = p.FunctionalCausalNode((0,), p.FrozenLinearFunction(((3.0, 0.5),)))
    if family == "functional":
        base = p.FunctionalCausalRecipe(task, (root, child), ((0, 0), (0, 1)), ((1, 0),), limits)
    elif family == "native":
        base = p.NativeCausalRecipe(
            task,
            (p.NativeCausalNode(root), p.NativeCausalNode(child)),
            ((0, 0), (0, 1)),
            (1, 0),
            limits,
        )
    elif family == "vector":
        base = p.NativeVectorRecipe(
            task,
            (
                p.NativeVectorNode(root, (p.VariableDomain(), p.VariableDomain())),
                p.NativeVectorNode(child, (p.VariableDomain(),)),
            ),
            ((0, 0), (0, 1)),
            (1, 0),
            limits,
        )
    else:
        base = p.ConvertedCausalRecipe(
            task,
            (p.ConvertedCausalNode(root), p.ConvertedCausalNode(child)),
            (p.ConvertedReadout(0, channel=0), p.ConvertedReadout(0, channel=1)),
            p.ConvertedReadout(1, channel=0),
            limits,
        )
    return p.GPRootDAGRecipe(source, base, (p.GPRootBinding(0, (0, 1)),), limits)


def fit(recipe: p.GPRootDAGRecipe, **kwargs: Any) -> p.GPRootCalibration:
    return p.calibrate_gp_root_dag(
        recipe, calibration_seed=73, calibration_task_id=2**55 + 19, **kwargs
    )


@pytest.mark.parametrize("coherent", [False, True])
def test_complete_future_fit_rejects_before_cholesky(
    coherent: bool, monkeypatch: pytest.MonkeyPatch
):
    cap = p.CausalLimits(working_bytes=120 * 1024**2)
    original = make_original(cap if coherent else p.CausalLimits())

    def forbidden(*args: object, **kwargs: object):
        pytest.fail("known aggregate/cap mismatch reached covariance work")

    monkeypatch.setattr(np.linalg, "cholesky", forbidden)
    with pytest.raises(ValueError, match="GP.*(aggregate|limits)"):
        fit(original, calibration_rows=32, limits=cap)


@pytest.mark.parametrize("family", ["functional", "native", "vector", "converted"])
@pytest.mark.parametrize(
    ("target", "rows", "fit_rows", "location"),
    [
        ("linear", 16, 32, "uniform-r0"),
        ("linear", 16, 32, "grid-r0"),
        ("quadratic", 32, 40, "gaussian-r0"),
        ("neural", 48, 32, "fixed-r0"),
    ],
)
def test_varied_calibration_keeps_literal_statistics(
    family: str, target: str, rows: int, fit_rows: int, location: str
):
    original = make_original(
        p.CausalLimits(), family=family, rows=rows, target=target, location=location
    )
    kwargs: dict[str, Any] = {"calibration_rows": fit_rows}
    if location == "fixed-r0":
        kwargs["calibration_locations"] = tuple(
            (float(i) + 0.25, float(i % 3) + 0.5) for i in range(fit_rows)
        )
    actual = fit(original, **kwargs)
    values = np.asarray(actual.fit.source.draw.values)
    raw = 3 * values[:, 0] + 0.5 * values[:, 1]
    np.testing.assert_allclose(actual.calibration.center, [raw.mean()], rtol=0, atol=1e-12)
    np.testing.assert_allclose(actual.calibration.scale, [raw.std()], rtol=0, atol=1e-12)
    restored = p.recipe_from_dict(asdict(actual))
    assert asdict(restored) == asdict(actual)


def scalar_record_cost(value: Any) -> tuple[int, int]:
    if isinstance(value, dict):
        parts = [
            scalar_record_cost(v) for pair in cast(dict[str, Any], value).items() for v in pair
        ]
    elif isinstance(value, (list, tuple)):
        parts = [scalar_record_cost(v) for v in cast(list[Any], value)]
    elif value is None:
        return 0, 0
    elif isinstance(value, str):
        return 0, len(value.encode())
    else:
        return 1, 64
    return sum(n for n, _ in parts), 32 + sum(b for _, b in parts)


@pytest.mark.parametrize("field", ["working_bytes", "parameters"])
def test_exact_complete_envelope_boundary(field: str):
    original = make_original(p.CausalLimits())
    reference = fit(original, calibration_rows=32)
    raw = asdict(reference)
    raw["calibration"]["constant_channels"] = (0,)
    numbers, storage = scalar_record_cost(raw)
    reports = [
        original.source.request.admit(),
        reference.fit.source.request.admit(),
        original.base.admit(),
        reference.fit.base.admit(),
    ]
    minimum = (
        8 * storage + 4 * sum(r["estimated_working_bytes"] for r in reports)
        if field == "working_bytes"
        else numbers + sum(r["parameters"] for r in reports)
    )
    for delta in (-1, 0, 1):
        caps = p.CausalLimits(**{field: minimum + delta})
        source = make_original(caps)
        calls: list[int] = []
        cholesky = np.linalg.cholesky

        def seen(*args: Any, _calls: list[int] = calls, _cholesky: Any = cholesky, **kwargs: Any):
            _calls.append(1)
            return _cholesky(*args, **kwargs)

        with patch.object(np.linalg, "cholesky", seen):
            if delta < 0:
                with pytest.raises(ValueError, match="aggregate"):
                    fit(source, calibration_rows=32)
                assert calls == []
            else:
                actual = fit(source, calibration_rows=32)
                assert calls
                assert actual.calibration.center == reference.calibration.center


def test_nested_population_cap_cannot_be_overridden(monkeypatch: pytest.MonkeyPatch):
    source = make_original(p.CausalLimits(working_bytes=120 * 1024**2))

    def forbidden(*args: object, **kwargs: object):
        pytest.fail("nested aggregate cap reached covariance work")

    monkeypatch.setattr(np.linalg, "cholesky", forbidden)
    with pytest.raises(ValueError, match="aggregate"):
        fit(source, calibration_rows=32, limits=p.CausalLimits())


@pytest.mark.parametrize("limit", [False, 0, {}, "cpu"])
def test_invalid_cap_types_reject_before_replay(limit: Any, monkeypatch: pytest.MonkeyPatch):
    original = make_original(p.CausalLimits())

    def forbidden(*args: object, **kwargs: object):
        pytest.fail("invalid cap type reached covariance work")

    monkeypatch.setattr(np.linalg, "cholesky", forbidden)
    with pytest.raises(ValueError, match="limits"):
        fit(original, calibration_rows=32, limits=limit)
