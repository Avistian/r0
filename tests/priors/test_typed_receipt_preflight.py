"""Receipt metadata rejects before original sampled custody is replayed."""

import copy
from dataclasses import asdict
from typing import Any

import numpy as np
import pytest

from r0.modules import priors as p


@pytest.fixture(scope="module")
def retained_policy():
    profile = p.sample_source_root_profile(
        p.SourceRootProfile(1, selector="forced-r0", families=("gaussian",)),
        model_rng=np.random.default_rng(11),
        selection_rng=np.random.default_rng(13),
    )
    root = p.sample_source_root_base(profile, modes=("identity",))
    base = p.FunctionalCausalRecipe(
        p.LinearRecipe(rows=16, context_rows=8, features=1, signal_features=1, active_features=1),
        (
            p.FunctionalCausalNode(
                (), p.FrozenLinearFunction(((1.0,),)), root, version="functional-causal-node-v2"
            ),
            p.FunctionalCausalNode((0,), p.FrozenLinearFunction(((2.0,),))),
        ),
        ((0, 0),),
        ((1, 0),),
    )
    request = p.TypedInterventionPolicyRequest(
        base,
        (1,),
        (p.InterventionPolicyLocation(0, channels=(0,)),),
        eligibility=(0,),
        count=1,
        reference_rows=16,
    )
    policy = p.sample_typed_intervention_policy(
        request,
        selection_rng=np.random.default_rng(3),
        value_rng=np.random.default_rng(5),
        calibration_rng=np.random.default_rng(7),
    )
    return policy, type(profile.initial_rng[0])


@pytest.mark.parametrize(
    "case",
    [
        "initial-empty",
        "final-empty",
        "snapshot-version",
        "snapshot-engine",
        "snapshot-digest",
        "snapshot-payload",
        "snapshot-environment",
        "selected-empty",
        "selected-bool",
        "selected-ineligible",
        "bounds-empty",
        "bounds-width",
        "bounds-order",
        "bounds-nonfinite",
        "bounds-version",
        "reference-bool",
        "raw-values",
        "raw-dtype",
        "raw-extra",
        "raw-sequence",
        "raw-stream",
        "runtime-mode",
        "runtime-population",
        "runtime-base",
        "runtime-empty",
        "runtime-location",
        "runtime-outside-bound",
        "bounds-location",
        "final-engine-mismatch",
    ],
)
def test_receipt_preflight(retained_policy: Any, monkeypatch: pytest.MonkeyPatch, case: str):
    policy, snapshot = retained_policy
    raw = copy.deepcopy(asdict(policy))
    if case == "initial-empty":
        raw["initial"] = []
    elif case == "final-empty":
        raw["final"] = []
    elif case.startswith("snapshot-"):
        field, value = {
            "version": ("version", "invalid"),
            "engine": ("engine", "invalid"),
            "digest": ("digest", "bad"),
            "payload": ("payload", "invalid"),
            "environment": ("numpy_version", "invalid"),
        }[case[9:]]
        raw["final"][0][field] = value
    elif case == "selected-empty":
        raw["selected"] = []
    elif case == "selected-bool":
        raw["selected"] = [False]
    elif case == "selected-ineligible":
        raw["selected"] = [1]
    elif case == "bounds-empty":
        raw["bounds"] = []
    elif case == "bounds-width":
        raw["bounds"][0]["minimum"] = []
    elif case == "bounds-order":
        raw["bounds"][0]["minimum"] = [1e30]
    elif case == "bounds-nonfinite":
        raw["bounds"][0]["maximum"] = [float("inf")]
    elif case == "bounds-version":
        raw["bounds"][0]["version"] = "invalid"
    elif case == "reference-bool":
        raw["reference_seed"] = True
    elif case == "raw-values":
        raw["raw"][0]["values"] = []
    elif case == "raw-dtype":
        raw["raw"][0]["dtype"] = "<f4"
    elif case == "raw-extra":
        raw["raw"][0]["extra"] = 1
    elif case == "raw-sequence":
        raw["raw"][0]["sequence"] = 99
    elif case == "raw-stream":
        raw["raw"][0]["stream"] = "unknown"
    elif case == "runtime-mode":
        raw["runtime"]["mode"] = "matched"
    elif case == "runtime-population":
        raw["runtime"]["population"] += 1
    elif case == "runtime-base":
        raw["runtime"]["base"]["task"]["rows"] = 17
    elif case == "runtime-empty":
        raw["runtime"]["interventions"] = []
    elif case == "runtime-location":
        raw["runtime"]["interventions"][0]["location"] = "readout"
    elif case == "runtime-outside-bound":
        raw["runtime"]["interventions"][0]["values"] = (1e30,)
    elif case == "bounds-location":
        raw["bounds"][0]["location"] = True
    elif case == "final-engine-mismatch":
        raw["final"][0]["engine"] = "Philox"
    calls: list[int] = []
    original = snapshot.restore

    def tracked(self: Any, *args: Any, **kwargs: Any):
        calls.append(1)
        return original(self, *args, **kwargs)

    monkeypatch.setattr(snapshot, "restore", tracked)
    with pytest.raises(ValueError, match="."):
        p.recipe_from_dict(raw)
    assert calls == []
