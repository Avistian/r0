"""Policies retain typed topology construction while intervening on its runtime."""

import json
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import pytest
import torch
from test_typed_parent_topology import composed_request

from r0.modules import priors as p
from r0.modules.generation import load_causal_pair, save_causal_pair


@pytest.mark.parametrize("family", ["functional", "native", "native-vector", "converted"])
@pytest.mark.parametrize("kind", ["hard", "soft"])
@pytest.mark.parametrize("mode", ["matched", "population"])
def test_typed_topology_policy_keeps_full_origin(family: str, kind: str, mode: str, tmp_path: Path):
    controls = replace(
        composed_request(family, "extension", "sum"),
        topology=p.StructuralTopologyRequest(p.AttachmentGraphPrior(4, edges_per_node=2)),
    )
    streams = [np.random.default_rng(141 + i) for i in range(5)]
    base = p.sample_typed_topology_graph(
        controls,
        topology_rng=streams[0],
        named_rng=streams[1],
        raw_rng=streams[2],
        calibration_rng=streams[3],
        fit_rng=streams[4],
    )
    graph = base.runtime
    targets = (
        (graph.target.node,)
        if isinstance(graph, p.ConvertedCausalRecipe)
        else (graph.target[0],)
        if isinstance(graph, (p.NativeCausalRecipe, p.NativeVectorRecipe))
        else tuple(sorted({i for i, _ in graph.target}))
    )
    node = next(i for i in range(len(graph.nodes)) if i not in targets)
    equation = (
        graph.nodes[node]
        if isinstance(graph, p.FunctionalCausalRecipe)
        else graph.nodes[node].equation
    )
    path = (
        ("matrix",)
        if isinstance(equation.function, p.FrozenLinearFunction)
        else ("children", "0", "matrix")
    )
    address = (
        p.InterventionPolicyLocation(node, "native-variable", variable=0, location="native-value")
        if family in ("native", "native-vector")
        else p.InterventionPolicyLocation(node, channels=(0,))
    )
    request = p.TypedInterventionPolicyRequest(
        base,
        targets,
        locations=(address,) if kind == "hard" else (),
        kind=kind,
        eligibility=(node,),
        count=1,
        mode=mode,
        parameter_laws=(p.InterventionParameterLaw(node, path),) if kind == "soft" else (),
        bounds_lifecycle="source-reference-population"
        if mode == "matched"
        else "independent-calibration-r0",
        reference_seed=211,
        reference_task_id=41,
        reference_rows=16,
        limits=controls.limits,
        raw_limits=controls.raw_limits,
    )
    policy = p.sample_typed_intervention_policy(
        request,
        selection_rng=np.random.default_rng(3),
        value_rng=np.random.default_rng(5),
        calibration_rng=np.random.default_rng(7),
    )
    assert asdict(policy.request.base) == asdict(base)
    restored = p.recipe_from_dict(json.loads(json.dumps(asdict(policy))), limits=controls.limits)
    assert asdict(restored) == asdict(policy)
    result = p.generate_task(policy, seed=211, task_id=41)
    # The retained envelope changes provenance, not the intervention equations.
    direct = p.CausalInterventionRecipe(
        graph, policy.runtime.interventions, mode, policy.runtime.population
    )
    expected = p.generate_task(direct, seed=211, task_id=41)
    torch.testing.assert_close(result.inputs, expected.inputs, rtol=0, atol=0)
    torch.testing.assert_close(result.targets, expected.targets, rtol=0, atol=0)

    if kind == "hard" and mode == "matched":
        pair = p.generate_typed_policy_pair(policy, seed=211, task_id=41)
        destination = tmp_path / "pair"
        save_causal_pair(destination, pair, max_bytes=20_000_000, limits=controls.limits)
        loaded = load_causal_pair(destination, limits=controls.limits)
        assert loaded.after.recipe_json == result.recipe_json
        torch.testing.assert_close(loaded.after.inputs, result.inputs, rtol=0, atol=0)
        assert p.explain_generation(result, [7, 2, 7], limits=controls.limits)["rows"] == [7, 2, 7]


def test_policy_rejects_underbudget_topology_before_restore(monkeypatch: pytest.MonkeyPatch):
    from typing import Any

    from r0.modules.priors.lib.generator_snapshot import GeneratorSnapshot

    controls = composed_request("functional", "extension", "sum")
    streams = [np.random.default_rng(141 + i) for i in range(5)]
    base = p.sample_typed_topology_graph(
        controls,
        topology_rng=streams[0],
        named_rng=streams[1],
        raw_rng=streams[2],
        calibration_rng=streams[3],
        fit_rng=streams[4],
    )
    graph = base.runtime
    assert isinstance(graph, p.FunctionalCausalRecipe)
    targets = tuple(sorted({i for i, _ in graph.target}))
    node = next(i for i in range(len(graph.nodes)) if i not in targets)
    request = p.TypedInterventionPolicyRequest(
        base,
        targets,
        (p.InterventionPolicyLocation(node, channels=(0,)),),
        eligibility=(node,),
        count=1,
        limits=controls.limits,
        raw_limits=controls.raw_limits,
    )
    raw = asdict(request)
    raw["base"]["request"]["raw_limits"]["numbers"] = 1

    def forbidden(*args: Any, **kwargs: Any):
        raise AssertionError("restored source RNG before admission")

    monkeypatch.setattr(GeneratorSnapshot, "restore", forbidden)
    with pytest.raises(ValueError, match="limit|support|budget"):
        p.typed_intervention_policy_request_from_dict(raw, limits=controls.limits)
