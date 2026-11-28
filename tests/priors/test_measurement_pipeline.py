"""Ordered measurement arithmetic, draw laws and transactional failure checks."""

import copy
import math
from dataclasses import replace

import numpy as np
import pytest

from r0.modules import priors as p
from r0.modules.priors.lib.generator_snapshot import snapshot_generator


@pytest.mark.parametrize(
    "kind", ["affine", "signed-power", "signed-log1p", "tanh", "sin", "exp", "clip", "round"]
)
def test_deterministic_maps(kind: str) -> None:
    values = (-2.5, -0.5, 0.0, 0.5, 2.5)
    x = np.array([[v, 99.0] for v in values])
    parameters = {
        "affine": (("scale", -2.0), ("shift", 1.0)),
        "signed-power": (("power", 0.5),),
        "clip": (("lower", -1.0), ("upper", 1.0)),
    }.get(kind, ())
    stage = p.MeasurementStage(kind, (0,), parameters)
    result = p.draw_measurements((stage,), x, rng=np.random.default_rng(30))
    expected: list[float] = []
    for v in values:
        match kind:
            case "affine":
                y = -2 * v + 1
            case "signed-power":
                y = math.copysign(math.sqrt(abs(v)), v)
            case "signed-log1p":
                y = math.copysign(math.log1p(abs(v)), v)
            case "tanh":
                y = math.tanh(v)
            case "sin":
                y = math.sin(v)
            case "exp":
                y = math.exp(v)
            case "clip":
                y = max(-1.0, min(1.0, v))
            case _:
                y = float(round(v))
        expected.append(y)
    np.testing.assert_allclose(result.values[:, 0], expected, rtol=1e-14, atol=1e-14)
    np.testing.assert_array_equal(result.values[:, 1], x[:, 1])
    assert result.initial == result.final


@pytest.mark.parametrize(
    "kind",
    [
        "gaussian-noise",
        "shared-gaussian-noise",
        "heteroscedastic-noise",
        "outliers",
        "student-noise",
    ],
)
@pytest.mark.parametrize(
    "engine",
    [np.random.PCG64, np.random.PCG64DXSM, np.random.Philox, np.random.SFC64, np.random.MT19937],
)
def test_noise_draw_oracles(kind: str, engine: type[np.random.BitGenerator]) -> None:
    x = np.array([[-2.0, 1.0], [0.0, 3.0], [2.0, 5.0]])
    oracle = np.random.Generator(engine(30))
    if kind == "shared-gaussian-noise":
        expected = x + oracle.normal(size=(3, 1))
    elif kind == "heteroscedastic-noise":
        expected = x + (1 + np.abs(x)) * oracle.normal(size=x.shape)
    elif kind == "outliers":
        active = oracle.random(x.shape) < 0.01
        expected = x + active * 10 * oracle.normal(size=x.shape)
    elif kind == "student-noise":
        expected = x + oracle.standard_t(3, size=x.shape)
    else:
        expected = x + oracle.normal(size=x.shape)
    result = p.draw_measurements(
        (p.MeasurementStage(kind, (0, 1)),), x, rng=np.random.Generator(engine(30))
    )
    np.testing.assert_allclose(result.values, expected, rtol=1e-14, atol=1e-14)
    assert result.final == snapshot_generator(oracle)
    replay = p.draw_measurements(result.stages, x, rng=result.initial.restore())
    np.testing.assert_array_equal(replay.values, result.values)
    assert replay.binding == result.binding


def test_order_is_semantic_and_no_values_are_silently_clipped() -> None:
    x = np.array([[3.0], [-3.0], [0.0]])
    clip = p.MeasurementStage("clip", (0,), (("lower", -0.5), ("upper", 0.5)))
    noise = p.MeasurementStage("gaussian-noise", (0,))
    first = p.draw_measurements((clip, noise), x, rng=np.random.default_rng(30))
    second = p.draw_measurements((noise, clip), x, rng=np.random.default_rng(30))
    normal = np.random.default_rng(30).normal(size=x.shape)
    np.testing.assert_allclose(first.values, np.clip(x, -0.5, 0.5) + normal)
    np.testing.assert_allclose(second.values, np.clip(x + normal, -0.5, 0.5))
    assert first.binding != second.binding
    assert np.any(np.abs(first.values) > 0.5)
    assert np.all(np.abs(second.values) <= 0.5)


def test_late_failure_rolls_back_all_stage_draws() -> None:
    rng = np.random.default_rng(30)
    before = copy.deepcopy(rng.bit_generator.state)
    x = np.ones((3, 1))
    stages = (
        p.MeasurementStage("gaussian-noise", (0,)),
        p.MeasurementStage("affine", (0,), (("shift", 1000.0),)),
        p.MeasurementStage("exp", (0,)),
    )
    with pytest.raises(ValueError, match="nonfinite"):
        p.draw_measurements(stages, x, rng=rng)
    assert rng.bit_generator.state == before
    np.testing.assert_array_equal(x, np.ones((3, 1)))
    with pytest.raises(ValueError, match="workspace"):
        p.draw_measurements(stages, x, rng=rng, limits=replace(p.CausalLimits(), working_bytes=1))
    assert rng.bit_generator.state == before


@pytest.mark.parametrize("probability", [0.0, 1.0])
def test_outlier_endpoints_keep_declared_draws(probability: float) -> None:
    x = np.zeros((3, 2))
    stage = p.MeasurementStage("outliers", (0, 1), (("probability", probability),))
    oracle = np.random.default_rng(30)
    oracle.random(x.shape)
    normal = oracle.normal(size=x.shape)
    result = p.draw_measurements((stage,), x, rng=np.random.default_rng(30))
    np.testing.assert_array_equal(result.values, probability * 10 * normal)
    assert result.final == snapshot_generator(oracle)


def test_invalid_or_unused_stage_parameters_rejected() -> None:
    with pytest.raises(ValueError, match="unknown"):
        p.MeasurementStage("sin", (0,), (("scale", 2.0),))
    with pytest.raises(ValueError, match="support"):
        p.MeasurementStage("clip", (0,), (("lower", 2.0), ("upper", 1.0)))
    with pytest.raises(ValueError, match="duplicate"):
        p.MeasurementStage("affine", (0,), (("scale", 1.0), ("scale", 2.0)))
