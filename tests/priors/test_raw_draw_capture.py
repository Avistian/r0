"""Construction retains original draws and rolls ledger/RNG back together."""

import copy
from dataclasses import asdict

import numpy as np
import pytest

from r0.modules import priors


@pytest.mark.parametrize(
    "family", ["gaussian", "importance", "factorized", "kernel", "transformed"]
)
def test_matrix_capture_preserves_equation_and_rng_with_lossless_kernel_draws(family: str) -> None:
    context = priors.JointSamplingContext(np.random.default_rng(28), mode="local")
    plan = priors.sample_joint_matrix(
        context,
        "matrix",
        rows=2,
        columns=3,
        batch=2,
        prior=priors.JointMatrixPrior(families=(family,), simple_families=("kernel",)),
    )
    decisions = context.freeze()
    rng = np.random.default_rng(19)
    control = copy.deepcopy(rng)
    original = priors.materialize_joint_matrix(plan, decisions, rng=control)
    capture = priors.RawDrawLedger()
    result = priors.materialize_joint_matrix(plan, decisions, rng=rng, capture=capture)
    assert asdict(result) == asdict(original)
    np.testing.assert_array_equal(rng.random(8), control.random(8))
    records = capture.freeze()
    assert records
    assert [r.sequence for r in records] == list(range(len(records)))
    replay = np.random.default_rng(19)
    for record in records:
        np.testing.assert_array_equal(record.replay(replay), record.array())
    # The transcript consumes exactly the materializer's original random stream.
    control = np.random.default_rng(19)
    priors.materialize_joint_matrix(plan, decisions, rng=control)
    np.testing.assert_array_equal(replay.random(8), control.random(8))
    if family in ("kernel", "transformed"):
        raw = next(r for r in records if r.role == "kernel-sign-normals")
        assert raw.distribution == "normal"
        assert np.any(np.abs(raw.array()) != 1)
        matrices = result.matrices
        if family == "transformed":
            children: list[priors.NormalizedWeights] = []
            for matrix in matrices:
                assert isinstance(matrix.source, priors.TransformedWeights)
                children.append(matrix.source.child)
            matrices = tuple(children)
        for index, matrix in enumerate(matrices):
            assert isinstance(matrix.source, priors.LaplaceKernelWeights)
            np.testing.assert_array_equal(np.sign(raw.array()[index]), matrix.source.signs)
        assert any(r.role == "point-innovations" for r in records)


def test_capture_budget_failure_rolls_back_rng_and_existing_records() -> None:
    context = priors.JointSamplingContext(np.random.default_rng(28), mode="local")
    plan = priors.sample_joint_matrix(
        context,
        "matrix",
        rows=2,
        columns=3,
        batch=2,
        prior=priors.JointMatrixPrior(families=("gaussian",)),
    )
    rng = np.random.default_rng(19)
    control = copy.deepcopy(rng)
    capture = priors.RawDrawLedger(limits=priors.RawDrawLimits(numbers=12))
    # The first 12-value draw fits; the final normalization jitter does not.
    with pytest.raises(ValueError, match="raw draw.*limit"):
        priors.materialize_joint_matrix(plan, context.freeze(), rng=rng, capture=capture)
    assert capture.freeze() == ()
    np.testing.assert_array_equal(rng.random(16), control.random(16))


def test_capture_rejects_unbound_rng_without_advancing_it() -> None:
    capture = priors.RawDrawLedger()
    primary = np.random.default_rng(1)
    foreign = np.random.default_rng(2)
    control = copy.deepcopy(foreign)
    with capture.transaction(primary) as (_, staged), pytest.raises(ValueError, match="bound"):
        staged.draw(foreign, owner=("node",), role="value", distribution="normal", shape=(3,))
    assert capture.freeze() == ()
    np.testing.assert_array_equal(foreign.random(8), control.random(8))


def test_failed_raw_replay_preserves_callers_original_rng() -> None:
    from dataclasses import replace

    capture = priors.RawDrawLedger()
    rng = np.random.default_rng(3)
    with capture.transaction(rng) as (staged_rng, staged):
        staged.draw(staged_rng, owner=("node",), role="value", distribution="normal", shape=(3,))
    record = capture.freeze()[0]
    corrupt = replace(record, values=(100.0, *record.values[1:]))
    replay = np.random.default_rng(3)
    control = copy.deepcopy(replay)
    with pytest.raises(ValueError, match="replay differs"):
        corrupt.replay(replay)
    np.testing.assert_array_equal(replay.random(8), control.random(8))


@pytest.mark.parametrize("position", ["before", "after"])
def test_capture_rejects_unrecorded_draws_in_staged_rng(position: str) -> None:
    capture = priors.RawDrawLedger()
    rng = np.random.default_rng(3)
    control = copy.deepcopy(rng)

    def attempt() -> None:
        with capture.transaction(rng) as (staged_rng, staged):
            if position == "before":
                staged_rng.normal()
            staged.draw(
                staged_rng, owner=("node",), role="value", distribution="normal", shape=(3,)
            )
            if position == "after":
                staged_rng.normal()

    with pytest.raises(ValueError, match="unrecorded"):
        attempt()
    assert capture.freeze() == ()
    np.testing.assert_array_equal(rng.random(8), control.random(8))


@pytest.mark.parametrize("engine", ["PCG64", "PCG64DXSM", "Philox", "SFC64", "MT19937"])
def test_capture_replays_all_supported_bit_generators(engine: str) -> None:
    rng = np.random.Generator(getattr(np.random, engine)(31))
    replay = copy.deepcopy(rng)
    capture = priors.RawDrawLedger()
    with capture.transaction(rng) as (staged_rng, staged):
        staged.draw(staged_rng, owner=("node",), role="value", distribution="normal", shape=(7,))
        staged.draw(
            staged_rng,
            owner=("node",),
            role="order",
            distribution="permutation",
            shape=(7,),
            parameters=(("count", 7),),
        )
    for record in capture.freeze():
        np.testing.assert_array_equal(record.replay(replay), record.array())
    np.testing.assert_array_equal(rng.random(8), replay.random(8))


def test_later_capture_failure_preserves_prior_records_and_rng_position() -> None:
    context = priors.JointSamplingContext(np.random.default_rng(28), mode="local")
    plan = priors.sample_joint_matrix(
        context,
        "matrix",
        rows=2,
        columns=3,
        batch=2,
        prior=priors.JointMatrixPrior(families=("gaussian",)),
    )
    capture = priors.RawDrawLedger(limits=priors.RawDrawLimits(numbers=36))
    rng = np.random.default_rng(31)
    priors.materialize_joint_matrix(plan, context.freeze(), rng=rng, capture=capture)
    before = capture.freeze()
    control = copy.deepcopy(rng)
    assert len(before) == 2
    with pytest.raises(ValueError, match="raw draw.*limit"):
        priors.materialize_joint_matrix(plan, context.freeze(), rng=rng, capture=capture)
    assert capture.freeze() == before
    np.testing.assert_array_equal(rng.random(8), control.random(8))
