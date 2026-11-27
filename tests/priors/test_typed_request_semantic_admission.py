"""Complete outer semantics must reject before original sampled custody is replayed."""

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


@pytest.mark.parametrize("archive", [False, True])
@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("version", "invalid"),
        ("profile", "invalid"),
        ("kind", "invalid"),
        ("mode", "invalid"),
        ("bounds_lifecycle", "invalid"),
        ("reference_seed", True),
        ("reference_task_id", 1.0),
        ("population", False),
        ("population", 0),
        ("population", 2**63),
        ("fraction", 0),
        ("fraction", True),
        ("fraction", float("inf")),
        ("count", 2),
        ("count", True),
        ("count", 0),
        ("label_nodes", [0]),
        ("label_nodes", [1, 1]),
        ("label_nodes", [True]),
        ("eligibility", [0, 0]),
        ("eligibility", [2]),
        ("eligibility", [False]),
        ("locations", []),
        ("kind", "soft"),
        ("reference_rows", True),
        ("reference_rows", 0),
    ],
)
def test_invalid_outer_request_rejects_before_sampled_replay(
    retained_policy: Any, monkeypatch: pytest.MonkeyPatch, archive: bool, field: str, value: Any
):
    policy, snapshot = retained_policy
    data = copy.deepcopy(asdict(policy) if archive else asdict(policy.request))
    request = data["request"] if archive else data
    request[field] = value
    calls: list[int] = []
    restore = snapshot.restore

    def observed(self: Any, *args: Any, **kwargs: Any):
        calls.append(1)
        return restore(self, *args, **kwargs)

    monkeypatch.setattr(snapshot, "restore", observed)
    decode = p.recipe_from_dict if archive else p.typed_intervention_policy_request_from_dict
    with pytest.raises(
        ValueError,
        match="typed|policy|reference|channel|native|capture|parameter|metadata|finite|eligible",
    ):
        decode(data)
    assert calls == [], "invalid outer semantics restored original sampled custody"


@pytest.mark.parametrize("archive", [False, True])
@pytest.mark.parametrize(
    "case", ["channel", "target", "native", "capture", "probability", "source_rows", "soft_path"]
)
def test_invalid_address_and_law_semantics_precede_replay(
    retained_policy: Any, monkeypatch: pytest.MonkeyPatch, archive: bool, case: str
):
    policy, snapshot = retained_policy
    data = copy.deepcopy(asdict(policy) if archive else asdict(policy.request))
    request = data["request"] if archive else data
    if case == "source_rows":
        request.update(bounds_lifecycle="source-reference-population", reference_rows=17)
    elif case == "soft_path":
        request.update(
            kind="soft",
            locations=(),
            parameter_laws=(asdict(p.InterventionParameterLaw(0, ("not_a_parameter",))),),
        )
    else:
        address = dict(request["locations"][0])
        if case == "channel":
            address["channels"] = (1,)
        if case == "target":
            address["node"], request["eligibility"] = 1, (1,)
        if case == "native":
            address.update(kind="native-variable", channels=(), variable=0, location="native-value")
        if case == "capture":
            address.update(kind="capture", channels=(), capture="missing", location="capture")
        if case == "probability":
            address["probabilities"] = (1.0,)
        request["locations"] = (address,)
    calls: list[int] = []
    restore = snapshot.restore

    def observed(self: Any, *args: Any, **kwargs: Any):
        calls.append(1)
        return restore(self, *args, **kwargs)

    monkeypatch.setattr(snapshot, "restore", observed)
    decode = p.recipe_from_dict if archive else p.typed_intervention_policy_request_from_dict
    with pytest.raises(
        ValueError,
        match="typed|policy|reference|channel|native|capture|parameter|metadata|finite|eligible",
    ):
        decode(data)
    assert calls == []


def test_valid_complete_request_and_receipt_remain_exact(retained_policy: Any):
    policy, _ = retained_policy
    assert asdict(p.typed_intervention_policy_request_from_dict(asdict(policy.request))) == asdict(
        policy.request
    )
    assert asdict(p.recipe_from_dict(asdict(policy))) == asdict(policy)


def test_nested_sampled_prototype_preview_preserves_valid_custody(
    retained_policy: Any, monkeypatch: pytest.MonkeyPatch
):
    from dataclasses import replace

    policy, snapshot = retained_policy
    original = policy.request.base
    bank = p.sample_prototype_root(
        p.PrototypeRootPlan(original.nodes[0].root, 16, 0.25),
        rng=np.random.default_rng(31),
    )
    base = replace(
        original,
        nodes=(replace(original.nodes[0], root=p.PrototypeRootBase(bank)), original.nodes[1]),
    )
    request = replace(policy.request, base=base)
    assert asdict(p.typed_intervention_policy_request_from_dict(asdict(request))) == asdict(request)
    raw = asdict(request)
    raw["count"] = 2

    def forbidden(*args: Any, **kwargs: Any):
        raise AssertionError("invalid outer request replayed nested prototype custody")

    monkeypatch.setattr(snapshot, "restore", forbidden)
    with pytest.raises(ValueError, match="eligibility/count"):
        p.typed_intervention_policy_request_from_dict(raw)
