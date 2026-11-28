"""Every retained joint envelope is validated before any construction RNG restore."""

import copy
from dataclasses import asdict
from typing import Any
from unittest.mock import patch

import numpy as np
import pytest

from r0.modules import priors as p


@pytest.fixture(scope="module")
def retained() -> p.JointHyperpriorGraph:
    prior = p.JointFunctionPrior("lin", matrix=p.JointMatrixPrior(families=("gaussian",)))
    request = p.JointHyperpriorGraphRequest(
        p.LinearRecipe(rows=16, context_rows=8, features=1, signal_features=1, active_features=1),
        p.FixedSizeTopologyHyperprior(4),
        calibration_rows=24,
        root_prior=p.JointRootPrior(("gaussian",), prior),
        function_prior=prior,
        limits=p.CausalLimits(parameters=8_000_000, working_bytes=2 * 1024**3),
    )
    streams = [np.random.default_rng(i + 71) for i in range(5)]
    return p.sample_joint_hyperprior_graph(
        request,
        topology_rng=streams[0],
        named_rng=streams[1],
        raw_rng=streams[2],
        calibration_rng=streams[3],
        fit_rng=streams[4],
    )


@pytest.mark.parametrize(
    ("path", "value"),
    [
        (("runtime", "task", "rows"), True),
        (("runtime", "task", "rows"), 17),
        (("runtime", "observed"), ((1, 0),)),
        (("runtime", "nodes", 0, "parents"), (0,)),
        (("runtime", "nodes", 0, "function", "matrix"), ((2.0,),)),
        (("plan", "calibration_rows"), True),
        (("plan", "nodes", 0, "width"), 2),
        (("owners", 0, "reference"), ("wrong",)),
        (("registry", "operations", 0, "operation"), ("wrong",)),
        (("admission", "planned_parameters"), True),
        (("raw", 0, "sequence"), True),
        (("raw", 0, "shape"), (True,)),
        (("raw", 0, "generator"), "invalid"),
        (("raw", 0, "stream"), "named"),
        (("raw", 0, "state_after"), "0" * 64),
        (("owners", 1, "reference"), ("nodes", False)),
        (("runtime", "nodes", 0, "exogenous", "scale"), 1),
        (("realizations", 0, "calibration", "base", "width"), True),
        (("realizations", 0, "outputs"), ((0.0,),)),
        (
            ("calibrations",),
            (
                {
                    "node": True,
                    "identity": "x",
                    "parents": (),
                    "raw_start": 0,
                    "raw_stop": 1,
                    "input_digest": "0" * 64,
                    "version": "graph-calibration-receipt-v1",
                },
            ),
        ),
    ],
)
def test_nested_metadata_rejects_before_any_restore(
    retained: p.JointHyperpriorGraph,
    path: tuple[str | int, ...],
    value: Any,
):
    raw = asdict(retained)
    destination = raw["graph"]
    for key in path[:-1]:
        destination = destination[key]
    destination[path[-1]] = value
    with (
        patch.object(
            type(retained.topology.initial),
            "restore",
            side_effect=AssertionError("metadata reached RNG restoration"),
        ),
        pytest.raises(ValueError, match="."),
    ):
        p.joint_hyperprior_graph_from_dict(raw, limits=retained.limits)


@pytest.mark.parametrize(
    "field",
    [
        "runtime",
        "plan",
        "registry",
        "owners",
        "realizations",
        "calibrations",
        "raw",
        "admission",
        "initial",
        "final",
    ],
)
def test_required_graph_fields_cannot_be_defaulted(retained: p.JointHyperpriorGraph, field: str):
    raw = asdict(retained)
    del raw["graph"][field]
    with (
        patch.object(
            type(retained.topology.initial),
            "restore",
            side_effect=AssertionError("missing metadata reached RNG"),
        ),
        pytest.raises(ValueError, match="."),
    ):
        p.joint_hyperprior_graph_from_dict(raw, limits=retained.limits)


def test_injected_sampled_function_rejected_without_replay(retained: p.JointHyperpriorGraph):
    sampled = p.sample_feature_conv1d(
        p.FeatureConv1DRequest(1, 1, (p.Conv1DLayerPrior((p.Conv1DGeometry(1, 1),)),)),
        named_rng=np.random.default_rng(17),
        raw_rng=np.random.default_rng(19),
    )
    raw = asdict(retained)
    raw["graph"]["runtime"]["nodes"][0]["function"] = asdict(sampled)
    with (
        patch.object(
            type(retained.topology.initial),
            "restore",
            side_effect=AssertionError("injected sampled source restored"),
        ),
        pytest.raises(ValueError, match="sampled envelopes"),
    ):
        p.joint_hyperprior_graph_from_dict(raw, limits=retained.limits)


def test_complete_valid_archive_unchanged(retained: p.JointHyperpriorGraph):
    raw = asdict(retained)
    before = copy.deepcopy(raw)
    restored = p.joint_hyperprior_graph_from_dict(raw, limits=retained.limits)
    assert asdict(restored) == before == raw


@pytest.mark.parametrize("location", ["topology", "initial", "final"])
@pytest.mark.parametrize("mutation", ["digest", "state", "missing", "whitespace"])
def test_snapshot_state_is_checked_before_first_restore(
    retained: p.JointHyperpriorGraph,
    location: str,
    mutation: str,
):
    import json

    raw = asdict(retained)
    snapshot = raw["topology"]["initial"] if location == "topology" else raw["graph"][location][2]
    state = json.loads(snapshot["payload"])
    if mutation == "digest":
        snapshot["digest"] = "0" * 64
    elif mutation == "state":
        state["state"]["state"] = True
        snapshot["payload"] = json.dumps(state, separators=(",", ":"))
    elif mutation == "missing":
        del state["has_uint32"]
        snapshot["payload"] = json.dumps(state, separators=(",", ":"))
    else:
        snapshot["payload"] = json.dumps(state, indent=2)
    with (
        patch.object(
            type(retained.topology.initial),
            "restore",
            side_effect=AssertionError("malformed snapshot reached restore"),
        ),
        pytest.raises(ValueError, match="."),
    ):
        p.joint_hyperprior_graph_from_dict(raw, limits=retained.limits)
