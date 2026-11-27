"""Typed policy workflows retain their original graphs, exact identities and saved pairs."""

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from r0.modules import priors as p
from r0.modules.explorer import ExplorerSession
from r0.modules.generation import load_causal_pair


def vector_base() -> p.NativeVectorRecipe:
    continuous = p.VariableDomain()
    category = p.VariableDomain("categorical", ("red", "green", "blue"), "softmax")
    root = p.NativeVectorNode(
        p.FunctionalCausalNode(
            (),
            p.FrozenLinearFunction(((1.0,), (1.0,), (-1.0,), (0.0,))),
            p.JointRootBase("gaussian", 1),
        ),
        (continuous, category),
    )
    child = p.NativeVectorNode(
        p.FunctionalCausalNode((0,), p.FrozenLinearFunction(((2.0, 3.0, 5.0, 7.0),))),
        (continuous,),
    )
    return p.NativeVectorRecipe(
        p.LinearRecipe(rows=16, context_rows=8, features=2, signal_features=2, active_features=2),
        (root, child),
        ((0, 0), (0, 1)),
        (1, 0),
    )


def draft() -> dict[str, Any]:
    return {
        "profile": "explicit-typed-policy-r0-v1",
        "streams": ["18014398509482001", "18014398509482003", "18014398509482007"],
        "kind": "hard",
        "eligibility": [0],
        "count": 1,
        "mode": "matched",
        "reference_rows": 16,
        "locations": [
            {
                "node": 0,
                "kind": "native-variable",
                "variable": 1,
                "location": "native-value",
                "probabilities": [0.0, 0.0, 1.0],
            }
        ],
    }


def test_typed_category_pair_reopen_trace_frozen_next_and_rollback(tmp_path: Path) -> None:
    session = ExplorerSession(tmp_path)
    base = vector_base()
    baseline = session.request(
        "generate",
        {"recipe": asdict(base), "seed": "18014398509481989", "task_id": "18014398509481991"},
    )
    result = session.request("sample_intervention_policy", {"policy_json": json.dumps(draft())})
    assert result["recipe_json"] == baseline["recipe_json"]
    saved = load_causal_pair(tmp_path / result["intervention"]["path"])
    np.testing.assert_array_equal(saved.before.inputs[:, 0], saved.after.inputs[:, 0])
    np.testing.assert_array_equal(saved.after.inputs[:, 1], 2)
    np.testing.assert_allclose(saved.after.targets, 2 * saved.before.inputs[:, 0] + 7)
    record = json.loads(saved.after.recipe_json)
    assert record["task_id"] == 18014398509481991
    assert record["recipe"]["request"]["base"] == json.loads(json.dumps(asdict(base)))
    assert not record["recipe"]["reference_raw"]
    trace = session.request(
        "intervention_graph",
        {"indices": [15, 0, 15], "expected_intervention_id": result["intervention"]["after_id"]},
    )
    assert trace["after"]["rows"] == [15, 0, 15]
    assert any(n["id"] == "policy-realization" for n in trace["after"]["nodes"])
    for invalid in (
        {"policy_json": "{}"},
        {"policy_json": json.dumps(draft()), "max_seconds": 0.001},
    ):
        with pytest.raises(ValueError, match="failed"):
            session.request("sample_intervention_policy", invalid)
        assert session.request("view", {}) == result
    reopened = session.request("open_intervention", {"path": result["intervention"]["path"]})
    assert reopened["intervention"]["after_id"] == result["intervention"]["after_id"]
    active = session.request(
        "open",
        {"archive": result["intervention"]["path"] + "/after.zip", "task_id": "18014398509481991"},
    )
    assert session.request("graph", {"indices": [15, 0, 15]})["graph"]["rows"] == [15, 0, 15]
    advanced = session.request(
        "generate",
        {
            "record_json": active["recipe_json"],
            "frozen_identity": {"seed": "18014398509481989", "task_id": "18014398509481992"},
        },
    )
    assert json.loads(advanced["recipe_json"])["recipe"] == record["recipe"]


def test_public_request_decoder_admits_before_original_sampled_restore(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
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
    assert asdict(p.typed_intervention_policy_request_from_dict(asdict(request))) == asdict(request)

    session = ExplorerSession(tmp_path)
    baseline = session.request("generate", {"recipe": asdict(base), "seed": 19, "task_id": 7})

    def forbidden(*args: Any, **kwargs: Any):
        raise AssertionError("sampled base restored before request admission")

    monkeypatch.setattr(type(profile.initial_rng[0]), "restore", forbidden)
    with pytest.raises(ValueError, match="limit"):
        p.typed_intervention_policy_request_from_dict(
            asdict(request), limits=p.CausalLimits(working_bytes=1024**2)
        )
    # Worker subprocesses import an unpatched class. Parent-side replay would fail.
    draft_record = {
        "profile": "explicit-typed-policy-r0-v1",
        "streams": [3, 5, 7],
        "kind": "hard",
        "eligibility": [0],
        "count": 1,
        "reference_rows": 16,
        "locations": [{"node": 0, "channels": [0]}],
    }
    created = session.request(
        "sample_intervention_policy", {"policy_json": json.dumps(draft_record)}
    )
    assert created["recipe_json"] == baseline["recipe_json"]
    graph = session.request(
        "intervention_graph",
        {"indices": [15, 0, 15], "expected_intervention_id": created["intervention"]["after_id"]},
    )
    assert graph["after"]["rows"] == [15, 0, 15]


@pytest.mark.parametrize("family", ["normal-r0", "uniform-r0", "log-uniform-r0", "choice-r0"])
def test_typed_soft_laws_literal_values_and_descendants(tmp_path: Path, family: str) -> None:
    session = ExplorerSession(tmp_path)
    base = vector_base()
    baseline = session.request("generate", {"recipe": asdict(base), "seed": 19, "task_id": 7})
    policy = {
        "profile": "explicit-typed-policy-r0-v1",
        "streams": [3, 5, 7],
        "kind": "soft",
        "mode": "matched",
        "eligibility": [0],
        "count": 1,
        "parameter_laws": [
            {
                "node": 0,
                "path": ["matrix"],
                "family": family,
                "location": 0.25,
                "scale": 2.0,
                "choices": [-2.0, 4.0] if family == "choice-r0" else [],
            }
        ],
    }
    created = session.request("sample_intervention_policy", {"policy_json": json.dumps(policy)})
    pair = load_causal_pair(tmp_path / created["intervention"]["path"])
    receipt = json.loads(pair.after.recipe_json)["recipe"]
    assert receipt["request"]["base"] == json.loads(baseline["recipe_json"])["recipe"]
    rng = np.random.default_rng(5)
    if family == "normal-r0":
        expected = 0.25 + 2 * rng.normal(size=(4, 1))
    elif family == "uniform-r0":
        expected = 0.25 + 2 * rng.random(size=(4, 1))
    elif family == "log-uniform-r0":
        expected = np.exp(np.log(0.25) + rng.random(size=(4, 1)) * (np.log(2) - np.log(0.25)))
    else:
        expected = np.asarray([-2.0, 4.0])[np.floor(rng.random(size=(4, 1)) * 2).astype(int)]
    actual = receipt["runtime"]["interventions"][0]["function"]["matrix"]
    np.testing.assert_allclose(actual, expected, rtol=1e-14, atol=0)
    np.testing.assert_allclose(pair.after.inputs[:, 0], pair.before.inputs[:, 0] * expected[0, 0])
    categories = np.asarray(pair.after.inputs[:, 1], dtype=int)
    np.testing.assert_allclose(
        pair.after.targets,
        2 * np.asarray(pair.after.inputs[:, 0]) + np.asarray([3, 5, 7])[categories],
    )
