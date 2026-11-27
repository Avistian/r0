"""Named operation ownership includes deterministic and event-free child plans."""

import json
from dataclasses import asdict
from typing import Any

import numpy as np
import pytest

from r0.modules import priors


def test_event_free_uniform_weight_operation_cannot_reuse_a_completed_path() -> None:
    context = priors.JointSamplingContext(np.random.default_rng(28))
    plan = priors.sample_joint_weights(context, "weights", width=2, family="uniform")
    assert plan.operation == ("weights",)
    assert context.freeze().events == ()
    with pytest.raises(ValueError, match="already completed"):
        priors.sample_joint_weights(context, "weights", width=3, family="uniform")


def test_registry_retains_preorder_empty_scopes_and_exact_event_spans() -> None:
    context = priors.JointSamplingContext(np.random.default_rng(28), mode="local")
    with context.transaction("graph") as graph:
        priors.sample_joint_weights(graph, "uniform", width=2, family="uniform")
        with graph.transaction("deterministic"):
            pass
        with graph.transaction("node") as node:
            node.numerical("scale", priors.NumericalParameter("scale", 0.0, 1.0))
        graph.numerical("last", priors.NumericalParameter("offset", 0.0, 1.0))
    registry = context.freeze_registry()
    assert isinstance(registry, priors.JointOperationRegistry)
    assert registry.decisions == context.freeze()
    assert [
        (span.operation, span.event_start, span.event_stop) for span in registry.operations
    ] == [
        (("graph",), 0, 2),
        (("graph", "uniform"), 0, 0),
        (("graph", "deterministic"), 0, 0),
        (("graph", "node"), 0, 1),
    ]


def test_operation_registry_roundtrips_event_free_scopes_without_changing_decisions() -> None:
    context = priors.JointSamplingContext(np.random.default_rng(28))
    with context.transaction("graph") as graph:
        priors.sample_joint_weights(graph, "empty", width=2, family="uniform")
        priors.sample_joint_weights(graph, "sampled", width=2, family="polynomial")
    record = context.freeze_registry()
    restored = priors.joint_operation_registry_from_dict(json.loads(json.dumps(asdict(record))))
    assert restored == record
    assert restored.decisions == context.freeze()


@pytest.mark.parametrize(
    "engine",
    [np.random.PCG64, np.random.PCG64DXSM, np.random.MT19937, np.random.Philox, np.random.SFC64],
)
def test_failed_nested_scope_rolls_back_registry_and_allows_same_path_retry(
    engine: type[np.random.BitGenerator],
) -> None:
    rng = np.random.Generator(engine(35))
    control = np.random.Generator(engine(35))
    context = priors.JointSamplingContext(rng)
    before = context.freeze_registry()
    with pytest.raises(ValueError, match="abort"), context.transaction("graph") as graph:  # noqa: PT012
        priors.sample_joint_weights(graph, "empty", width=2, family="uniform")
        priors.sample_joint_weights(graph, "sampled", width=2, family="polynomial")
        raise ValueError("abort")
    assert context.freeze_registry() == before
    with context.transaction("graph") as graph:
        priors.sample_joint_weights(graph, "empty", width=2, family="uniform")
        priors.sample_joint_weights(graph, "sampled", width=2, family="polynomial")
    reference = priors.JointSamplingContext(control)
    with reference.transaction("graph") as graph:
        priors.sample_joint_weights(graph, "empty", width=2, family="uniform")
        priors.sample_joint_weights(graph, "sampled", width=2, family="polynomial")
    assert context.freeze_registry() == reference.freeze_registry()
    np.testing.assert_array_equal(rng.random(10), control.random(10))


def test_registry_snapshots_reject_open_and_closed_child_handles() -> None:
    context = priors.JointSamplingContext(np.random.default_rng(28))
    with context.transaction("outer") as outer:
        with pytest.raises(ValueError, match="active child"):
            context.freeze_registry()
        with pytest.raises(ValueError, match="outer context after completion"):
            outer.freeze_registry()
    with pytest.raises(ValueError, match="closed"):
        outer.freeze_registry()
    assert len(context.freeze_registry().operations) == 1


def test_caught_child_failure_leaves_no_phantom_registry_owner() -> None:
    context = priors.JointSamplingContext(np.random.default_rng(28))
    with context.transaction("outer") as outer:
        with pytest.raises(ValueError, match="abort"), outer.transaction("child"):  # noqa: PT012
            raise ValueError("abort")
        with outer.transaction("child"):
            pass
    assert [span.operation for span in context.freeze_registry().operations] == [
        ("outer",),
        ("outer", "child"),
    ]


@pytest.mark.parametrize(
    "limits",
    [priors.JointRegistryLimits(operations=1), priors.JointRegistryLimits(path_bytes=1)],
)
def test_empty_operations_are_resource_bounded_without_consuming_rng(
    limits: priors.JointRegistryLimits,
) -> None:
    rng, control = np.random.default_rng(28), np.random.default_rng(28)
    context = priors.JointSamplingContext(rng, registry_limits=limits)
    with context.transaction("a"):
        pass
    before = context.freeze_registry()
    with pytest.raises(ValueError, match="registry limit"), context.transaction("b"):
        pass
    assert context.freeze_registry() == before
    np.testing.assert_array_equal(rng.random(10), control.random(10))


def _saved_registry() -> dict[str, Any]:
    context = priors.JointSamplingContext(np.random.default_rng(28), mode="local")
    with context.transaction("graph") as graph:
        priors.sample_joint_weights(graph, "empty", width=2, family="uniform")
        with graph.transaction("node") as node:
            node.numerical("scale", priors.NumericalParameter("scale", 0.0, 1.0))
        graph.numerical("last", priors.NumericalParameter("last", 0.0, 1.0))
    with context.transaction("tail"):
        pass
    return json.loads(json.dumps(asdict(context.freeze_registry())))


@pytest.mark.parametrize(
    "mutation",
    [
        "duplicate",
        "orphan",
        "missing-parent",
        "reordered",
        "foreign-event",
        "empty-span",
        "bool-index",
        "unknown",
        "missing",
        "version",
        "ancestor-span",
        "sibling-overlap",
    ],
)
def test_registry_decoder_rejects_inconsistent_operation_ownership(mutation: str) -> None:
    payload = _saved_registry()
    operations = payload["operations"]
    if mutation == "duplicate":
        operations.append(operations[1])
    elif mutation == "orphan":
        del operations[2]
    elif mutation == "missing-parent":
        del operations[0]
    elif mutation == "reordered":
        operations[1], operations[2] = operations[2], operations[1]
    elif mutation == "foreign-event":
        operations[2]["event_stop"] = 2
    elif mutation == "empty-span":
        operations[1]["event_stop"] = 1
    elif mutation == "bool-index":
        operations[0]["event_start"] = False
    elif mutation == "unknown":
        operations[0]["unexpected"] = 0
    elif mutation == "missing":
        del operations[0]["event_stop"]
    elif mutation == "version":
        payload["version"] = "future-registry"
    elif mutation == "ancestor-span":
        operations[0]["event_stop"] = 1
    else:
        operations[-1]["event_start"] = 0
        operations[-1]["event_stop"] = 0
    with pytest.raises(ValueError, match="registry|span|registered|preorder"):
        priors.joint_operation_registry_from_dict(payload)


def test_registry_decoder_enforces_caller_caps_even_with_larger_payload_caps() -> None:
    payload = _saved_registry()
    payload["limits"] = {"operations": 10**9, "path_bytes": 10**9}
    with pytest.raises(ValueError, match="registry limit"):
        priors.joint_operation_registry_from_dict(
            payload, registry_limits=priors.JointRegistryLimits(operations=2)
        )
    with pytest.raises(ValueError, match="registry limit"):
        priors.joint_operation_registry_from_dict(
            payload, registry_limits=priors.JointRegistryLimits(path_bytes=5)
        )
    with pytest.raises(ValueError, match="limit"):
        priors.joint_operation_registry_from_dict(payload, limits=priors.CausalLimits(parameters=2))


@pytest.mark.parametrize(("count", "path_bytes"), [(-1, 0), (0, -1), (True, 0), (0, True)])
def test_registry_admission_rejects_invalid_counters(count: int, path_bytes: int) -> None:
    with pytest.raises(ValueError, match="nonnegative integers"):
        priors.JointRegistryLimits().admit(count, path_bytes)
