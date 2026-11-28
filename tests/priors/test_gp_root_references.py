"""Original-row GP references keep hidden root coordinates and independent fit fixed."""

import hashlib
import json
from dataclasses import asdict

import numpy as np
import pytest

from r0.modules import priors as p


def source(kind: str):
    controls = p.LinearRecipe(
        rows=16, context_rows=8, features=2, signal_features=2, active_features=2, noise_ratio=0.2
    )
    root = p.FunctionalCausalNode(
        (),
        p.FrozenLinearFunction(((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))),
        p.JointRootBase("gaussian", 3),
        noise_scale=(0.0, 0.0, 0.3),
    )
    child = p.FunctionalCausalNode(
        (0,), p.FrozenLinearFunction(((2.0, -1.0, 4.0),)), noise_scale=(0.7,)
    )
    base = p.FunctionalCausalRecipe(controls, (root, child), ((0, 0), (0, 1)), ((1, 0),))
    population = p.realize_gp_columns(p.GPColumnRequest(rows=16, columns=3), seed=17, task_id=19)
    original = p.GPRootDAGRecipe(population, base, (p.GPRootBinding(0, (0, 1, 2)),))
    if kind == "plain":
        return original
    calibrated = p.calibrate_gp_root_dag(
        original, calibration_rows=32, calibration_seed=23, calibration_task_id=29
    )
    return (
        calibrated
        if kind == "calibrated"
        else p.realize_gp_root_calibration_query(calibrated, seed=31, task_id=37)
    )


def rng(seed: int, task: int, *keys: object):
    value = json.dumps(
        ("functional-causal-v1", seed, task, keys), sort_keys=True, separators=(",", ":")
    )
    return np.random.default_rng(int(hashlib.sha256(value.encode()).hexdigest(), 16))


@pytest.mark.parametrize("kind", ["plain", "calibrated", "query"])
@pytest.mark.parametrize("policy", ["zero", "original-row"])
def test_hidden_gp_root_reference_literal(kind: str, policy: str):
    original = source(kind)
    before = asdict(original)
    base = original if isinstance(original, p.GPRootDAGRecipe) else original.original
    row = 5
    points = ((1.0, 2.0), (-2.0, 0.5), (0.0, 0.0))
    seed, task = base.source.seed, base.source.task_id
    reference = p.evaluate_gp_root_reference(
        original, row=row, inputs=points, structural_innovations=policy
    )
    expected = np.asarray([2 * a - b + 4 * base.source.draw.values[row][2] for a, b in points])
    if policy == "original-row":
        expected += (
            1.2 * rng(seed, task, "innovation", 0, 2).normal(size=16)[row]
            + 0.7 * rng(seed, task, "innovation", 1, 0).normal(size=16)[row]
        )
    np.testing.assert_allclose(np.asarray(reference.raw_scores)[:, 0], expected, atol=1e-12)
    calibrated = (
        expected
        if isinstance(original, p.GPRootDAGRecipe)
        else (expected - original.calibration.center[0]) / original.calibration.scale[0]
    )
    np.testing.assert_allclose(np.asarray(reference.scores)[:, 0], calibrated, atol=1e-12)
    assert reference.target_mean is not None
    np.testing.assert_allclose(reference.target_mean, calibrated, atol=1e-12)
    np.testing.assert_allclose(
        reference.targets,
        calibrated + 0.2 * rng(seed, task, "noise").normal(size=16)[row],
        atol=1e-12,
    )
    assert asdict(original) == before
    assert json.loads(reference.source_json) == json.loads(json.dumps(before))
    assert p.gp_root_reference_from_dict(json.loads(json.dumps(asdict(reference)))) == reference


@pytest.mark.parametrize("family", ["functional", "native", "vector", "converted"])
@pytest.mark.parametrize("kind", ["plain", "calibrated", "query"])
def test_all_families_reference_fixed_fit(family: str, kind: str):
    from gp_root_fixture import gp_root_recipe

    original = gp_root_recipe(family)
    retained = original
    if kind != "plain":
        retained = p.calibrate_gp_root_dag(
            original, calibration_rows=32, calibration_seed=101, calibration_task_id=103
        )
        if kind == "query":
            retained = p.realize_gp_root_calibration_query(retained, seed=107, task_id=109)
    reference = p.evaluate_gp_root_reference(
        retained, row=9, inputs=((1.0, -2.0), (3.0, 0.5)), structural_innovations="zero"
    )
    raw = np.asarray([4.0, 5.5])
    expected = (
        raw
        if isinstance(retained, p.GPRootDAGRecipe)
        else (raw - retained.calibration.center[0]) / retained.calibration.scale[0]
    )
    np.testing.assert_allclose(reference.targets, expected, atol=1e-12)
    assert p.gp_root_reference_from_dict(json.loads(json.dumps(asdict(reference)))) == reference


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("seed", True),
        ("task_id", 19.0),
        ("population_id", "wrong"),
        ("source_fingerprint", "wrong"),
        ("fit_population_id", "wrong"),
        ("raw_scores", ((0.0,),)),
        ("scores", ((0,), (0,))),
        ("targets", (0.0,)),
        ("classes", ("bad",)),
        ("feature_schema", ()),
        ("semantics", "wrong"),
    ],
)
def test_reference_malformed_receipt_before_replay(
    field: str, value: object, monkeypatch: pytest.MonkeyPatch
):
    reference = p.evaluate_gp_root_reference(
        source("plain"), row=5, inputs=((1.0, 2.0), (2.0, 3.0))
    )
    raw = asdict(reference)
    raw[field] = value

    def forbidden(*args: object, **kwargs: object):
        pytest.fail("malformed receipt began GP replay")

    monkeypatch.setattr(np.linalg, "cholesky", forbidden)
    with pytest.raises(
        ValueError, match="GP|reference|limit|budget|rows|metadata|number|bytes|sizes"
    ):
        p.gp_root_reference_from_dict(raw)


@pytest.mark.parametrize(
    "case", ["row", "float-row", "nan", "width", "policy", "capture", "budget", "depth"]
)
def test_reference_controls_reject_before_source_replay(case: str, monkeypatch: pytest.MonkeyPatch):
    from typing import Any

    retained = source("query")
    kwargs: dict[str, Any] = {"row": 5, "inputs": ((1.0, 2.0),)}
    if case == "row":
        kwargs["row"] = 16
    if case == "float-row":
        kwargs["row"] = 5.0
    if case == "nan":
        kwargs["inputs"] = ((float("nan"), 2.0),)
    if case == "width":
        kwargs["inputs"] = ((1.0,),)
    if case == "policy":
        kwargs["structural_innovations"] = "resample"
    if case == "capture":
        kwargs["capture_location"] = "interpolate"
    if case == "budget":
        kwargs["limits"] = p.CausalLimits(working_bytes=1024)
    raw = asdict(retained)
    if case == "depth":
        raw["original"]["base"]["task"]["rows"] = True

    def forbidden(*args: object, **kw: object):
        pytest.fail("invalid reference started replay")

    monkeypatch.setattr(np.linalg, "cholesky", forbidden)
    monkeypatch.setattr(np.random, "default_rng", forbidden)
    with pytest.raises(
        ValueError, match="GP|reference|limit|budget|rows|metadata|number|bytes|sizes"
    ):
        p.gp_root_reference_from_source(raw, **kwargs)
