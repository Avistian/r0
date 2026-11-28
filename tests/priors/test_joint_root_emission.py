"""Fresh root rows use the frozen mechanism without refitting calibration."""

import json
import pickle
from dataclasses import asdict

import numpy as np
import pytest
import torch

from r0.modules import priors


@pytest.mark.parametrize("family", ["uniform", "ball", "gaussian", "covariance"])
@pytest.mark.parametrize("function", ["lin", "quad", "gp", "tree", "disc", "em", "mlp", "prod"])
def test_joint_root_emits_fresh_rows_without_refitting(family: str, function: str) -> None:
    context = priors.JointSamplingContext(np.random.default_rng(114), mode="local")
    plan = priors.sample_joint_root(
        context,
        "root",
        width=2,
        calibration_rows=4,
        prior=priors.JointRootPrior(
            families=(family,),
            function=priors.JointFunctionPrior(
                functions=function,
                frequencies=4,
                matrix=priors.JointMatrixPrior(families=("gaussian",)),
            ),
        ),
    )
    root = priors.materialize_joint_root(
        plan,
        context.freeze(),
        rng=np.random.default_rng(115),
        calibration_rng=np.random.default_rng(116),
        fit_rng=np.random.default_rng(117),
        calibration_id="root-fit",
    )
    before = pickle.dumps(root)
    rng, oracle = np.random.default_rng(118), np.random.default_rng(118)
    emitted = priors.sample_joint_root_values(
        root, context.freeze(), rows=7, rng=rng, sample_id="fresh-rows"
    )
    assert isinstance(emitted, priors.JointRootEmission)
    assert emitted.root is root
    assert pickle.dumps(root) == before
    base = root.calibration.base
    uniform = family == "uniform" or base.covariance_base == "uniform"
    draws = oracle.random((7, 2)) if uniform else oracle.normal(size=(7, 2))
    expected = -1 + 2 * draws if uniform else draws
    if family == "ball":
        radii = oracle.random(7)
        np.testing.assert_array_equal(emitted.sample.radial_uniform, radii)
        expected = draws / np.linalg.norm(draws, axis=1)[:, None] * np.sqrt(radii[:, None])
    if family == "covariance":
        expected = expected @ np.asarray(base.factor).T
    np.testing.assert_array_equal(emitted.sample.draws, draws)
    np.testing.assert_allclose(emitted.sample.values, expected, rtol=1e-14, atol=1e-15)
    actual = priors.evaluate_frozen_function(
        root.function.function, torch.tensor(emitted.sample.values, dtype=torch.float64)
    )
    np.testing.assert_array_equal(emitted.outputs, actual.numpy())
    assert rng.bit_generator.state == oracle.bit_generator.state
    assert emitted.sample_id == "fresh-rows"
    assert len(emitted.outputs) == 7

    restored = priors.joint_root_emission_from_dict(
        json.loads(json.dumps(asdict(emitted))), decisions=context.freeze()
    )
    assert restored == emitted


def _linear_root(
    function: str = "lin", matrix: str = "gaussian"
) -> tuple[priors.JointRootRealization, priors.JointSamplingRecord]:
    context = priors.JointSamplingContext(np.random.default_rng(1), mode="local")
    plan = priors.sample_joint_root(
        context,
        "root",
        width=1,
        calibration_rows=3,
        prior=priors.JointRootPrior(
            families=("covariance",),
            function=priors.JointFunctionPrior(
                functions=function,
                matrix=priors.JointMatrixPrior(families=(matrix,), vector_family="uniform"),
            ),
        ),
    )
    decisions = context.freeze()
    root = priors.materialize_joint_root(
        plan,
        decisions,
        rng=np.random.default_rng(2),
        calibration_rng=np.random.default_rng(3),
        fit_rng=np.random.default_rng(4),
        calibration_id="root-fit",
    )
    assert root.plan.covariance_base == "gaussian"
    return root, decisions


@pytest.mark.parametrize("rows", [True, 0, -1, 10**9])
def test_root_emission_rejects_invalid_or_unaffordable_rows_before_rng_commit(rows: int) -> None:
    root, decisions = _linear_root()
    rng = np.random.default_rng(118)
    before = pickle.dumps(rng.bit_generator.state)
    with pytest.raises(ValueError, match="rows|limit"):
        priors.sample_joint_root_values(root, decisions, rows=rows, rng=rng, sample_id="bad-rows")
    assert pickle.dumps(rng.bit_generator.state) == before


def test_root_emission_real_overflow_rolls_back_rng_and_allows_valid_recovery() -> None:
    root, decisions = _linear_root()
    payload = json.loads(json.dumps(asdict(root)))
    base = payload["calibration"]["base"]
    base["gaussian_factor"] = [[1.7e308]]
    base["weights"]["draws"] = [0.9]
    base["factor"] = [[1.7e308 * 0.9]]
    payload["calibration"]["draws"] = [[0.0]] * 3
    payload["calibration"]["values"] = [[0.0]] * 3
    payload["outputs"] = [[0.0]] * 3
    extreme = priors.joint_root_realization_from_dict(payload, decisions=decisions)
    rng = np.random.default_rng(3)
    before = pickle.dumps(rng.bit_generator.state)
    with pytest.raises(ValueError, match="finite"):
        priors.sample_joint_root_values(extreme, decisions, rows=1, rng=rng, sample_id="overflow")
    assert pickle.dumps(rng.bit_generator.state) == before
    recovered = priors.sample_joint_root_values(
        root, decisions, rows=1, rng=rng, sample_id="recovery"
    )
    fresh = priors.sample_joint_root_values(
        root, decisions, rows=1, rng=np.random.default_rng(3), sample_id="recovery"
    )
    assert recovered == fresh


@pytest.mark.parametrize("field", ["draws", "values", "outputs", "base", "identity"])
def test_saved_root_emission_rejects_tampered_draws_outputs_or_binding(field: str) -> None:
    root, decisions = _linear_root()
    emitted = priors.sample_joint_root_values(
        root, decisions, rows=2, rng=np.random.default_rng(118), sample_id="fresh"
    )
    payload = json.loads(json.dumps(asdict(emitted)))
    if field == "outputs":
        payload["outputs"][0][0] += 0.25
    elif field == "identity":
        payload["sample_id"] = root.calibration_id
    elif field == "base":
        payload["sample"]["base"]["gaussian_factor"] = [[0.0]]
        payload["sample"]["base"]["factor"] = [[0.0]]
    else:
        payload["sample"][field][0][0] += 0.25
    with pytest.raises(ValueError, match="differ"):
        priors.joint_root_emission_from_dict(payload, decisions=decisions)


def test_root_emission_rejects_calibration_identity_before_consuming_rng() -> None:
    root, decisions = _linear_root()
    rng = np.random.default_rng(118)
    before = pickle.dumps(rng.bit_generator.state)
    with pytest.raises(ValueError, match="identity must differ"):
        priors.sample_joint_root_values(
            root, decisions, rows=2, rng=rng, sample_id=root.calibration_id
        )
    assert pickle.dumps(rng.bit_generator.state) == before


@pytest.mark.parametrize("function", ["mlp", "prod"])
@pytest.mark.parametrize(
    "matrix", ["gaussian", "importance", "factorized", "kernel", "transformed"]
)
def test_saved_root_emission_roundtrips_nested_function_and_matrix_families(
    function: str, matrix: str
) -> None:
    root, decisions = _linear_root(function, matrix)
    emitted = priors.sample_joint_root_values(
        root, decisions, rows=5, rng=np.random.default_rng(118), sample_id="nested-emission"
    )
    restored = priors.joint_root_emission_from_dict(
        json.loads(json.dumps(asdict(emitted))), decisions=decisions
    )
    assert restored == emitted
