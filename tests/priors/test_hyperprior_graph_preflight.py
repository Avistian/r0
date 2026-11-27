"""Outer topology graph metadata rejects before sampled template restoration."""

import copy
from dataclasses import asdict
from typing import Any

import numpy as np
import pytest

from r0.modules import priors as p


@pytest.fixture(scope="module")
def packet():
    profile = p.sample_source_root_profile(
        p.SourceRootProfile(1, selector="forced-r0", families=("gaussian",)),
        model_rng=np.random.default_rng(11),
        selection_rng=np.random.default_rng(13),
    )
    root = p.FunctionalCausalNode(
        (),
        p.FrozenLinearFunction(((1.0,),)),
        p.sample_source_root_base(profile, modes=("identity",)),
        version="functional-causal-node-v2",
    )
    edge = p.FunctionalCausalNode((0,), p.FrozenLinearFunction(((2.0,),)))
    request = p.HyperpriorGraphRequest(
        p.LinearRecipe(rows=16, context_rows=8, features=1, signal_features=1, active_features=1),
        p.FixedSizeTopologyHyperprior(2),
        root,
        edge,
    )
    recipe = p.sample_hyperprior_graph(request, rng=np.random.default_rng(31))
    return asdict(recipe), type(recipe.topology.initial)


@pytest.mark.parametrize(
    "case",
    [
        "layout",
        "sharing",
        "request-version",
        "outer-version",
        "root-parents",
        "edge-parents",
        "edge-width",
        "task-rows",
        "task-class-offsets",
        "topology-request",
        "topology-limits",
        "topology-raw-limits",
        "topology-version",
        "topology-algorithm",
        "initial-header",
        "final-header",
        "initial-json",
        "initial-engine",
        "final-digest",
        "topology-parents",
        "topology-components",
        "topology-gamma",
        "raw-values",
        "raw-shape",
        "raw-type",
    ],
)
def test_outer_metadata_rejects_before_restore(
    packet: Any, monkeypatch: pytest.MonkeyPatch, case: str
):
    raw, cls = packet
    raw = copy.deepcopy(raw)
    req = raw["request"]
    top = raw["topology"]
    if case == "layout":
        req["layout"] = "invalid"
    elif case == "sharing":
        req["mechanism_sharing"] = "invalid"
    elif case == "request-version":
        req["version"] = "invalid"
    elif case == "outer-version":
        raw["version"] = "invalid"
    elif case == "root-parents":
        req["root_template"]["parents"] = (0,)
    elif case == "edge-parents":
        req["edge_template"]["parents"] = (1,)
    elif case == "edge-width":
        req["edge_template"]["function"]["matrix"] = ((1.0, 2.0),)
    elif case == "task-rows":
        req["task"]["rows"] = True
    elif case == "task-class-offsets":
        req["task"]["class_offsets"] = ""
    elif case == "topology-request":
        top["request"]["gamma_rate"] = 3.0
    elif case == "topology-limits":
        top["limits"]["working_bytes"] *= 2
    elif case == "topology-raw-limits":
        top["raw_limits"]["numbers"] *= 2
    elif case == "topology-version":
        top["version"] = "invalid"
    elif case == "topology-algorithm":
        top["algorithm"] = "invalid"
    elif case == "initial-header":
        top["initial"]["version"] = "invalid"
    elif case == "final-header":
        top["final"]["numpy_version"] = "invalid"
    elif case == "initial-json":
        top["initial"]["payload"] = "invalid"
    elif case == "initial-engine":
        top["initial"]["engine"] = "invalid"
    elif case == "final-digest":
        top["final"]["digest"] = "bad"
    elif case == "topology-parents":
        top["parents"] = ()
    elif case == "topology-components":
        top["components"] = ()
    elif case == "topology-gamma":
        top["gamma_value"] = False
    elif case == "raw-values":
        top["raw"][0]["values"] = ()
    elif case == "raw-shape":
        top["raw"][0]["shape"] = (2,)
    elif case == "raw-type":
        top["raw"][0]["values"] = (True,)
    calls: list[int] = []
    restore = cls.restore

    def tracked(self: Any, *args: Any, **kwargs: Any):
        calls.append(1)
        return restore(self, *args, **kwargs)

    monkeypatch.setattr(cls, "restore", tracked)
    with pytest.raises(ValueError, match="."):
        p.hyperprior_graph_from_dict(raw)
    assert calls == []
