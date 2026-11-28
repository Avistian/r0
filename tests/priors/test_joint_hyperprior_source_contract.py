"""Retained source constants cannot change the joint sampler's authored contract."""

from dataclasses import asdict
from typing import Any, cast
from unittest.mock import patch

import numpy as np
import pytest

from r0.modules import priors as p


@pytest.fixture(scope="module")
def original() -> p.JointHyperpriorGraph:
    prior = p.JointFunctionPrior("lin", matrix=p.JointMatrixPrior(families=("gaussian",)))
    request = p.JointHyperpriorGraphRequest(
        p.LinearRecipe(rows=16, context_rows=8, features=1, signal_features=1, active_features=1),
        p.FixedSizeTopologyHyperprior(3),
        calibration_rows=24,
        root_prior=p.JointRootPrior(("gaussian",), prior),
        function_prior=prior,
        limits=p.CausalLimits(parameters=8_000_000, working_bytes=2 * 1024**3),
    )
    streams = [np.random.default_rng(101 + i) for i in range(5)]
    return p.sample_joint_hyperprior_graph(
        request,
        topology_rng=streams[0],
        named_rng=streams[1],
        raw_rng=streams[2],
        calibration_rng=streams[3],
        fit_rng=streams[4],
    )


@pytest.mark.parametrize("field", ["operations", "path_bytes"])
def test_fixed_registry_caps_reject_before_any_restore(
    original: p.JointHyperpriorGraph, field: str
):
    raw = asdict(original)
    raw["graph"]["registry"]["limits"][field] -= 1
    with (
        patch.object(
            type(original.topology.initial),
            "restore",
            side_effect=AssertionError("restore before source contract"),
        ),
        pytest.raises(ValueError, match="source|registry|contract"),
    ):
        p.recipe_from_dict(raw, limits=original.limits)


@pytest.mark.parametrize("field", ["max_draws", "max_values", "max_parameters"])
def test_fixed_preference_caps_reject_before_restore(original: p.JointHyperpriorGraph, field: str):
    raw = asdict(original)
    decisions = raw["graph"]["registry"]["decisions"]
    for record in (decisions, decisions["numerical"], decisions["categorical"]):
        record["limits"][field] -= 1
    with (
        patch.object(
            type(original.topology.initial),
            "restore",
            side_effect=AssertionError("restore before source contract"),
        ),
        pytest.raises(ValueError, match="source|contract|limit"),
    ):
        p.recipe_from_dict(raw, limits=original.limits)


@pytest.mark.parametrize("field", ["max_draws", "max_values", "max_parameters", "global_policy"])
def test_categorical_hyper_scope_retains_fixed_sampler_settings(
    original: p.JointHyperpriorGraph, field: str
):
    raw = asdict(original)
    hyper = raw["graph"]["registry"]["decisions"]["categorical"]["preferences"][0][
        "hyperparameters"
    ]
    if field == "global_policy":
        hyper[field] = "quantile"
    else:
        hyper["limits"][field] -= 1
    with (
        patch.object(
            type(original.topology.initial),
            "restore",
            side_effect=AssertionError("restore before source contract"),
        ),
        pytest.raises(ValueError, match="source|contract|limit"),
    ):
        p.recipe_from_dict(raw, limits=original.limits)


def test_global_policy_must_equal_source_request_before_restore(original: p.JointHyperpriorGraph):
    raw = asdict(original)
    decisions = raw["graph"]["registry"]["decisions"]
    decisions["global_policy"] = "strict-contract-v1"
    decisions["numerical"]["global_policy"] = "strict"
    with (
        patch.object(
            type(original.topology.initial),
            "restore",
            side_effect=AssertionError("restore before source contract"),
        ),
        pytest.raises(ValueError, match="source|contract|policy"),
    ):
        p.recipe_from_dict(raw, limits=original.limits)


def test_source_operation_root_is_fixed_before_replay(original: p.JointHyperpriorGraph):
    import json

    # Coherently rename path components, without touching version identifiers or keys.
    def rename(value: Any) -> Any:
        if isinstance(value, dict):
            return {k: rename(v) for k, v in cast(dict[str, Any], value).items()}
        if isinstance(value, list):
            return [rename(v) for v in cast(list[Any], value)]
        return "other" if value == "graph" else value

    raw = rename(json.loads(json.dumps(asdict(original))))
    with (
        patch.object(
            type(original.topology.initial),
            "restore",
            side_effect=AssertionError("restore before source contract"),
        ),
        pytest.raises(ValueError, match="source|contract"),
    ):
        p.recipe_from_dict(raw, limits=original.limits)


@pytest.mark.parametrize("mode", ["meta", "local", "global"])
@pytest.mark.parametrize("categorical_policy", ["shared", "refresh"])
@pytest.mark.parametrize("global_policy", ["scalar-quantile-v1", "strict-contract-v1"])
def test_source_contract_preserves_declared_policies(
    original: p.JointHyperpriorGraph, mode: str, categorical_policy: str, global_policy: str
):
    from dataclasses import replace

    request = replace(
        original.request,
        mode=mode,
        categorical_policy=categorical_policy,
        global_policy=global_policy,
    )
    streams = [np.random.default_rng(101 + i) for i in range(5)]
    recipe = p.sample_joint_hyperprior_graph(
        request,
        topology_rng=streams[0],
        named_rng=streams[1],
        raw_rng=streams[2],
        calibration_rng=streams[3],
        fit_rng=streams[4],
    )
    assert p.recipe_from_dict(asdict(recipe), limits=request.limits) == recipe


def test_plan_caps_match_source_request_before_restore(original: p.JointHyperpriorGraph):
    raw = asdict(original)
    raw["graph"]["plan"]["limits"]["nodes"] -= 1
    with (
        patch.object(
            type(original.topology.initial),
            "restore",
            side_effect=AssertionError("restore before source contract"),
        ),
        pytest.raises(ValueError, match="source|contract"),
    ):
        p.recipe_from_dict(raw, limits=original.limits)
