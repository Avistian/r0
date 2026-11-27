"""Shared sampling decisions keep one preference identity across owned operations."""

import copy
import json
from dataclasses import asdict, replace
from typing import Any

import numpy as np
import pytest

from r0.modules import priors


def test_joint_context_shares_named_preferences_across_operations_and_replays() -> None:
    context = priors.JointSamplingContext(np.random.default_rng(28))
    request = priors.NumericalParameter(
        "simple_random_weights_std_scale", 0.0001, 10.0, scale="log"
    )
    category = priors.CategoricalParameter("acts_torch", ("tanh", "relu", "sin"))
    for owner in ("matrix", "node", "converter"):
        with context.transaction(owner) as operation:
            operation.numerical("weight_scale", request)
            operation.categorical("activation", category)
    record = context.freeze()
    assert len(record.numerical.preferences) == 1
    assert len(record.categorical.preferences) == 1
    assert [draw.preference for draw in record.numerical.draws] == [0, 0, 0]
    assert [draw.preference for draw in record.categorical.draws] == [0, 0, 0]
    assert [event.operation for event in record.events] == [
        ("matrix",),
        ("matrix",),
        ("node",),
        ("node",),
        ("converter",),
        ("converter",),
    ]
    decoded = priors.joint_sampling_from_dict(json.loads(json.dumps(asdict(record))))
    assert decoded == record


def test_joint_nested_failure_rolls_back_and_closed_handles_cannot_draw() -> None:
    rng = np.random.Generator(np.random.MT19937(28))
    control = copy.deepcopy(rng)
    context = priors.JointSamplingContext(rng)
    request = priors.NumericalParameter("scale", 0.0, 1.0)
    before = context.freeze()
    child: priors.JointSamplingContext | None = None
    # Exercise mutations before rejecting the whole composite operation.
    with pytest.raises(ValueError, match="fit failed"), context.transaction("graph") as graph:  # noqa: PT012
        graph.numerical("first", request)
        with graph.transaction("child") as child:
            child.numerical("second", request)
        raise ValueError("fit failed")
    assert context.freeze() == before
    np.testing.assert_array_equal(rng.random(10), control.random(10))

    assert child is not None
    with pytest.raises(ValueError, match="closed"):
        child.numerical("stale", request)
    with context.transaction("graph") as graph:
        with pytest.raises(ValueError, match="active child"):
            context.freeze()
        with graph.transaction("child") as child:
            child.numerical("second", request)
            with pytest.raises(ValueError, match="active child"):
                graph.numerical("first", request)
        graph.numerical("last", request)
    assert [e.operation for e in context.freeze().events] == [("graph", "child"), ("graph",)]
    with pytest.raises(ValueError, match="closed"):
        graph.freeze()
    with pytest.raises(ValueError, match="already"), context.transaction("graph"):
        pass


def test_rejected_duplicate_role_does_not_consume_a_draw_when_caught() -> None:
    rng = np.random.default_rng(29)
    control = copy.deepcopy(rng)
    context = priors.JointSamplingContext(rng)
    reference = priors.JointSamplingContext(control)
    request = priors.NumericalParameter("scale", 0.0, 1.0)
    with context.transaction("node") as node:
        node.numerical("sigma", request)
        before = node.freeze()
        with pytest.raises(ValueError, match="role"):
            node.numerical("sigma", request)
        assert node.freeze() == before
        node.numerical("importance", request)
    with reference.transaction("node") as node:
        node.numerical("sigma", request)
        node.numerical("importance", request)
    assert context.freeze() == reference.freeze()
    np.testing.assert_array_equal(rng.random(10), control.random(10))


@pytest.mark.parametrize(
    "engine",
    [np.random.PCG64, np.random.PCG64DXSM, np.random.MT19937, np.random.Philox, np.random.SFC64],
)
def test_joint_commit_rejects_external_rng_mutation(engine: type[np.random.BitGenerator]) -> None:
    rng = np.random.Generator(engine(33))
    control = copy.deepcopy(rng)
    context = priors.JointSamplingContext(rng)
    before = context.freeze()
    # External RNG mutation is observed when the transaction exits.
    with pytest.raises(ValueError, match="RNG changed"), context.transaction("node") as node:  # noqa: PT012
        node.numerical("sigma", priors.NumericalParameter("sigma", 0.0, 1.0))
        assert rng.random() == control.random()
    assert context.freeze() == before
    np.testing.assert_array_equal(rng.random(10), control.random(10))


def test_joint_global_policy_ties_scalar_quantiles_and_rejects_law_changes() -> None:
    context = priors.JointSamplingContext(np.random.default_rng(0), mode="global")
    request = priors.NumericalParameter("decay", 0.0, 1.0)
    with context.transaction("graph") as graph:
        first = graph.numerical("first", request)
        second = graph.numerical("second", replace(request, low=2.0, high=4.0))
        assert first.units == second.units
        assert second.values == pytest.approx((2 + 2 * first.values[0],))
        before = graph.freeze()
        for incompatible in (
            replace(request, integer="round"),
            replace(request, boundary_mass=True),
            replace(request, low=0.1, scale="log"),
            replace(request, shape=(1,)),
        ):
            with pytest.raises(ValueError, match="contract|scalar"):
                graph.numerical("rejected", incompatible)
            assert graph.freeze() == before
    record = context.freeze()
    assert record.global_policy == "scalar-quantile-v1"
    assert record.numerical.global_policy == "quantile"
    strict = priors.JointSamplingContext(
        np.random.default_rng(0), mode="global", global_policy="strict-contract-v1"
    )
    vector = replace(request, shape=(2,))
    with strict.transaction("matrix") as matrix:
        first = matrix.numerical("first", vector)
        assert matrix.numerical("second", vector).values == first.values
        with pytest.raises(ValueError, match="contract"):
            matrix.numerical("third", replace(vector, high=2.0))


@pytest.mark.parametrize("cap", ["max_draws", "max_values", "max_parameters"])
def test_joint_limits_cover_both_samplers_and_nested_categorical_weights(cap: str) -> None:
    # Scalar meta numerical: 9 parameters. Three-choice meta categorical: 33.
    # Two event references add 2. Categorical weights own two additional draws.
    exact = priors.PreferenceLimits(max_draws=4, max_values=4, max_parameters=44)
    tighter = replace(exact, **{cap: getattr(exact, cap) - 1})
    rng = np.random.default_rng(44)
    control = copy.deepcopy(rng)
    request = priors.NumericalParameter("scale", 0.0, 1.0)
    choice = priors.CategoricalParameter("acts_torch", ("tanh", "relu", "sin"))
    context = priors.JointSamplingContext(rng, limits=tighter)
    # Each draw fits separately; rejection occurs only after combined usage.
    with pytest.raises(ValueError, match="limit"), context.transaction("node") as node:  # noqa: PT012
        node.numerical("scale", request)
        node.categorical("activation", choice)
    assert context.freeze().events == ()
    np.testing.assert_array_equal(rng.random(10), control.random(10))
    accepted = priors.JointSamplingContext(np.random.default_rng(44), limits=exact)
    with accepted.transaction("node") as node:
        node.numerical("scale", request)
        node.categorical("activation", choice)
    record = accepted.freeze()
    assert (record.draw_count, record.value_count, record.parameter_count) == (4, 4, 44)
    raw = asdict(record)
    raw["limits"] = asdict(tighter)
    # The aggregate preflight must precede child decoding and value validation.
    raw["numerical"]["draws"][0]["units"] = (float("nan"),)
    with pytest.raises(ValueError, match="limit"):
        priors.joint_sampling_from_dict(raw)


def test_forced_identity_is_owned_without_rng_or_selector_consumption() -> None:
    rng = np.random.default_rng(17)
    control = copy.deepcopy(rng)
    limits = priors.PreferenceLimits(max_draws=1, max_values=1, max_parameters=1)
    context = priors.JointSamplingContext(rng, limits=limits)
    with context.transaction("target") as target:
        event = target.forced_identity("converter", reason="target-warp-disabled")
        assert event.draw_index is None
        assert event.kind == "forced_identity"
        with pytest.raises(ValueError, match="limit"):
            target.forced_identity("second", reason="target-warp-disabled")
    record = context.freeze()
    assert record.numerical.draws == record.categorical.draws == ()
    assert (record.draw_count, record.value_count, record.parameter_count) == (1, 0, 0)
    np.testing.assert_array_equal(rng.random(10), control.random(10))
    raw = json.loads(json.dumps(asdict(record)))
    assert priors.joint_sampling_from_dict(raw) == record
    raw["events"][0]["reason"] = ""
    with pytest.raises(ValueError, match="reason"):
        priors.joint_sampling_from_dict(raw)


def test_joint_decode_rejects_reopened_operations_and_bounds_ownership_metadata() -> None:
    context = priors.JointSamplingContext(np.random.default_rng(1))
    with context.transaction("graph") as graph:
        with graph.transaction("first") as first:
            first.numerical("a", priors.NumericalParameter("scale", 0.0, 1.0))
        with graph.transaction("second") as second:
            second.numerical("b", priors.NumericalParameter("scale", 0.0, 1.0))
        graph.numerical("c", priors.NumericalParameter("scale", 0.0, 1.0))
    record = context.freeze()
    raw = json.loads(json.dumps(asdict(record)))
    raw["events"][2]["operation"] = ["graph", "first"]
    with pytest.raises(ValueError, match="reopen"):
        priors.joint_sampling_from_dict(raw)
    for bad_name in ("", " leading", "trailing ", "null\0name", "x" * 129, "é" * 65):
        with pytest.raises(ValueError, match="name"), context.transaction(bad_name):
            pass
    with pytest.raises(ValueError, match="depth"):
        replace(record.events[0], operation=("nested",) * 17)
    for field in ("role", "reason"):
        kwargs = {field: "x" * 129}
        event = priors.JointDecision(("node",), "identity", "forced_identity", reason="disabled")
        with pytest.raises(ValueError, match=field):
            replace(event, **kwargs)


@pytest.mark.parametrize("malformed", ["graph", {"graph": True}])
def test_joint_decoder_does_not_coerce_nonarray_operation_paths(malformed: Any) -> None:
    context = priors.JointSamplingContext(np.random.default_rng(128))
    with context.transaction("graph") as graph:
        graph.forced_identity("target", reason="target-warp-disabled")
    raw = json.loads(json.dumps(asdict(context.freeze())))
    raw["events"][0]["operation"] = malformed
    with pytest.raises(ValueError, match="array"):
        priors.joint_sampling_from_dict(raw)
