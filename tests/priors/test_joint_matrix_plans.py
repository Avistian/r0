"""Joint matrix branches retain source ordering and bound shared subplans."""

import copy
import json
from dataclasses import asdict

import numpy as np
import pytest
import torch
from numpy.typing import NDArray

from r0.modules import priors


@pytest.mark.parametrize(
    "family", ["gaussian", "importance", "factorized", "kernel", "transformed"]
)
def test_joint_matrix_families_bind_complete_source_branches(family: str) -> None:
    context = priors.JointSamplingContext(np.random.default_rng(28), mode="global")
    prior = priors.JointMatrixPrior(families=(family,), simple_families=("kernel",))
    plan = priors.sample_joint_matrix(
        context, "projection", rows=2, columns=3, batch=2, prior=prior
    )
    record = context.freeze()
    assert plan.family == family
    assert record.categorical.draws[0].request.name == "random_matrix_type"
    if family == "transformed":
        assert plan.child is not None
        assert plan.activation is not None
        assert plan.child.family == "kernel"
        assert plan.activation.selector == "plain"
        names = [d.request.name for d in record.categorical.draws]
        assert names[:3] == ["random_matrix_type", "simple_random_matrix_type", "act_type_plain"]
        operations = [e.operation for e in record.events]
        assert operations.index(("projection", "activation")) > max(
            i for i, operation in enumerate(operations) if operation[:2] == ("projection", "child")
        )
    kernel = plan.child if family == "transformed" else plan
    assert kernel is not None
    if kernel.family == "kernel":
        assert kernel.points is not None
        assert kernel.points.count == 10
        assert kernel.points.dimension == 3
        assert kernel.points.law == "gaussian-mixture"
        assert "random_kernel_matrix_dist_factor" in [
            d.request.name for d in record.numerical.draws
        ]
        assert all(d.request.name != "kernel_point_law" for d in record.categorical.draws)
    plan.validate(record)
    assert (
        priors.joint_matrix_plan_from_dict(json.loads(json.dumps(asdict(plan))), decisions=record)
        == plan
    )


@pytest.mark.parametrize("family", ["gaussian", "importance", "factorized"])
def test_joint_matrix_realization_uses_full_batch_source_order(family: str) -> None:
    context = priors.JointSamplingContext(np.random.default_rng(28), mode="local")
    plan = priors.sample_joint_matrix(
        context,
        "projection",
        rows=2,
        columns=3,
        batch=2,
        prior=priors.JointMatrixPrior(families=(family,), vector_family="uniform"),
    )
    rng = np.random.default_rng(17)
    control = copy.deepcopy(rng)
    realized = priors.materialize_joint_matrix(plan, context.freeze(), rng=rng)
    if family == "gaussian":
        raw = control.normal(size=(2, 2, 3))
    elif family == "importance":
        weights = control.random((4, 3)).reshape(2, 2, 3)
        raw = weights * control.normal(size=(2, 2, 3))
        raw /= 1e-6 + np.linalg.norm(raw, axis=-1, keepdims=True)
    else:
        left = control.normal(size=(2, 2, 2))
        weights = control.random((2, 2))
        right = control.normal(size=(2, 2, 3))
        raw = (left * weights[:, None, :]) @ right
    raw += 1e-6 * control.normal(size=(2, 2, 3))
    expected = raw / (1e-6 + np.linalg.norm(raw, axis=-1, keepdims=True))
    np.testing.assert_allclose(realized.materialize(), expected, rtol=1e-12, atol=1e-12)


def test_joint_transformed_batch_flattens_each_member_and_retains_three_jitters() -> None:
    context = priors.JointSamplingContext(np.random.default_rng(103), mode="local")
    plan = priors.sample_joint_matrix(
        context,
        "projection",
        rows=2,
        columns=3,
        batch=2,
        prior=priors.JointMatrixPrior(families=("transformed",), simple_families=("gaussian",)),
    )
    assert plan.activation is not None
    assert plan.activation.activation.kind == "softmax"
    rng = np.random.default_rng(17)
    control = copy.deepcopy(rng)
    realized = priors.materialize_joint_matrix(plan, context.freeze(), rng=rng)
    child = control.normal(size=(2, 2, 3)) + 1e-6 * control.normal(size=(2, 2, 3))
    child /= 1e-6 + np.linalg.norm(child, axis=-1, keepdims=True)
    flat = child.reshape(2, 6)
    exp = np.exp(flat - flat.max(axis=1, keepdims=True))
    raw = (exp / exp.sum(axis=1, keepdims=True)).reshape(2, 2, 3)
    raw += 1e-3 * control.normal(size=(2, 2, 3))
    raw += 1e-6 * control.normal(size=(2, 2, 3))
    expected = raw / (1e-6 + np.linalg.norm(raw, axis=-1, keepdims=True))
    np.testing.assert_allclose(realized.materialize(), expected, rtol=1e-12, atol=1e-12)
    np.testing.assert_array_equal(rng.random(8), control.random(8))
    np.testing.assert_array_equal(rng.random(8), control.random(8))


def test_joint_kernel_batch_retains_one_cloud_and_literal_laplace_arithmetic() -> None:
    context = priors.JointSamplingContext(np.random.default_rng(28), mode="local")
    plan = priors.sample_joint_matrix(
        context,
        "projection",
        rows=2,
        columns=3,
        batch=2,
        prior=priors.JointMatrixPrior(families=("kernel",)),
    )
    realized = priors.materialize_joint_matrix(
        plan, context.freeze(), rng=np.random.default_rng(17)
    )
    assert realized.point_cloud is not None
    assert plan.distance is not None
    assert realized.point_cloud.shape == (10, 3)
    points = np.asarray(realized.point_cloud.materialize()).reshape(2, 5, 3)
    expected: list[NDArray[np.float64]] = []
    for i, matrix in enumerate(realized.matrices):
        assert isinstance(matrix.source, priors.LaplaceKernelWeights)
        distances = np.linalg.norm(points[i, :2, None] - points[i, None, 2:], axis=-1)
        raw = np.exp(-plan.distance * distances) * np.asarray(matrix.source.signs)
        raw += 1e-6 * np.asarray(matrix.jitter)
        expected.append(raw / (1e-6 + np.linalg.norm(raw, axis=-1, keepdims=True)))
        assert matrix.source.point_construction is None
    np.testing.assert_allclose(realized.materialize(), expected, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize(
    "family", ["gaussian", "importance", "factorized", "kernel", "transformed"]
)
def test_joint_matrix_realizations_roundtrip_with_bound_geometry_and_limits(family: str) -> None:
    context = priors.JointSamplingContext(np.random.default_rng(28), mode="local")
    plan = priors.sample_joint_matrix(
        context,
        "projection",
        rows=2,
        columns=3,
        batch=2,
        prior=priors.JointMatrixPrior(families=(family,), simple_families=("kernel",)),
    )
    decisions = context.freeze()
    realized = priors.materialize_joint_matrix(plan, decisions, rng=np.random.default_rng(17))
    data = json.loads(json.dumps(asdict(realized)))
    restored = priors.joint_matrix_realization_from_dict(data, decisions=decisions)
    assert restored == realized
    assert restored.materialize() == realized.materialize()
    bad = copy.deepcopy(data)
    bad["matrices"][0]["jitter"] = {"0": [0.0, 0.0, 0.0], "1": [0.0, 0.0, 0.0]}
    with pytest.raises(ValueError, match="array"):
        priors.joint_matrix_realization_from_dict(bad, decisions=decisions)
    with pytest.raises(ValueError, match="limit"):
        priors.joint_matrix_realization_from_dict(
            data, decisions=decisions, limits=priors.CausalLimits(parameters=1)
        )
    construction = restored.matrices[0]
    values = construction.materialize()
    recipe = priors.CausalRecipe(
        task=priors.LinearRecipe(
            rows=16, context_rows=8, features=1, signal_features=1, active_features=1
        ),
        nodes=(
            priors.ActivatedCausalNode(bias=(1.0, -2.0), noise_scale=(0.0, 0.0)),
            priors.MatrixCausalNode(
                parents=(0,),
                weights=values,
                bias=(0.0, 0.0, 0.0),
                noise_scale=(0.0, 0.0, 0.0),
                weight_construction=construction,
            ),
        ),
        observed=((0, 0),),
        target=((1, 2),),
        version="causal-v2",
    )
    replay = priors.recipe_from_dict(json.loads(json.dumps(asdict(recipe))))
    task = priors.generate_task(replay, seed=28, task_id=22)
    torch.testing.assert_close(
        task.scores, torch.full((16, 1), values[0][2] - 2 * values[1][2], dtype=torch.float64)
    )
    assert priors.explain_generation(task, [0, 15])["target_max_error"] < 1e-12


def test_joint_matrix_rejects_after_partial_planning_and_at_exact_resource_boundaries() -> None:
    rng = np.random.default_rng(28)
    context = priors.JointSamplingContext(rng, mode="global", global_policy="strict-contract-v1")
    before = context.freeze()
    state = copy.deepcopy(rng.bit_generator.state)
    with pytest.raises(ValueError, match="contract"):
        priors.sample_joint_matrix(
            context,
            "rejected-kernel",
            rows=2,
            columns=3,
            prior=priors.JointMatrixPrior(
                families=("kernel",), scalarization="source-weight-batches-v1"
            ),
        )
    assert context.freeze() == before
    assert rng.bit_generator.state == state
    plan = priors.sample_joint_matrix(
        context,
        "gaussian",
        rows=2,
        columns=3,
        batch=2,
        prior=priors.JointMatrixPrior(families=("gaussian",)),
    )
    raw_rng = np.random.default_rng(17)
    state = copy.deepcopy(raw_rng.bit_generator.state)
    for limits in (priors.CausalLimits(parameters=33), priors.CausalLimits(working_bytes=6911)):
        with pytest.raises(ValueError, match="limit"):
            priors.materialize_joint_matrix(plan, context.freeze(), rng=raw_rng, limits=limits)
        assert raw_rng.bit_generator.state == state
    result = priors.materialize_joint_matrix(
        plan,
        context.freeze(),
        rng=raw_rng,
        limits=priors.CausalLimits(parameters=34, working_bytes=6912),
    )
    assert result.parameter_count == 28
