"""Structural topology variants preserve geometry archives and combined budgets."""

import copy
import json
from dataclasses import asdict, replace
from typing import cast

import numpy as np
import pytest

from r0.modules import priors
from r0.modules.episodes import TaskType


def _prior(family: str) -> priors.AttachmentGraphPrior | priors.MotifGraphPrior:
    if family == "attachment":
        return priors.AttachmentGraphPrior(nodes=4, edges_per_node=2, power=1.3)
    if family == "isolated-attachment":
        return priors.AttachmentGraphPrior(nodes=3, components=3)
    if family == "isolated-motif":
        return priors.MotifGraphPrior(depth=0, components=3)
    if family == "motif-rewrites":
        return priors.MotifGraphPrior(depth=2, rewrite_steps=3)
    return priors.MotifGraphPrior(depth=1, motifs=(family,))


def _request(family: str) -> priors.GeometryGraphRequest:
    functions = priors.JointFunctionPrior(
        functions="lin", matrix=priors.JointMatrixPrior(families=("gaussian",))
    )
    return priors.GeometryGraphRequest(
        topology=_prior(family),
        task=priors.LinearRecipe(
            rows=16, context_rows=8, features=2, signal_features=2, active_features=2
        ),
        width_prior=priors.JointWidthPrior(low=1, high=2),
        calibration_rows=4,
        root_prior=priors.JointRootPrior(families=("gaussian",), function=functions),
        function_prior=functions,
    )


def _sample(request: priors.GeometryGraphRequest) -> priors.SampledGeometryGraph:
    return priors.sample_geometry_graph(
        request,
        named_rng=np.random.default_rng(114),
        raw_rng=np.random.default_rng(115),
        calibration_rng=np.random.default_rng(116),
        fit_rng=np.random.default_rng(117),
    )


@pytest.mark.parametrize(
    "family",
    [
        "attachment",
        "chain",
        "fork",
        "collider",
        "motif-rewrites",
        "isolated-attachment",
        "isolated-motif",
    ],
)
def test_structural_topology_geometry_archive_generation_inspection_and_pairs(family: str) -> None:
    graph = _sample(_request(family))
    assert isinstance(graph.topology, priors.StructuralGraph)
    assert graph.replay() == graph
    archive = json.loads(json.dumps(asdict(graph)))
    original = copy.deepcopy(archive)
    assert priors.recipe_from_dict(archive) == graph
    assert archive == original
    emitted = priors.generate_task(graph, seed=19, task_id=7)
    assert priors.recipe_from_dict(json.loads(emitted.recipe_json)["recipe"]) == graph
    assert priors.explain_generation(emitted, [7, 0, 7])["rows"] == [7, 0, 7]
    resplit = graph.with_task(replace(graph.task, context_rows=4, split=1))
    other = priors.generate_task(resplit, seed=19, task_id=7)
    assert emitted.source_task_id == other.source_task_id
    assert emitted.episode_id != other.episode_id
    np.testing.assert_array_equal(emitted.inputs.numpy(), other.inputs.numpy())
    np.testing.assert_array_equal(emitted.targets.numpy(), other.targets.numpy())
    pair = priors.generate_pair(graph, intervention="noise", value=0.25, seed=19, task_id=7)
    assert all(json.loads(pair.card_json)["required_equalities"].values())
    changed = priors.paired_recipe(graph, intervention="noise", value=0.25)
    assert isinstance(changed, priors.SampledGeometryGraph)
    assert changed.topology == graph.topology
    assert changed.graph.raw == graph.graph.raw
    if family.startswith("isolated"):
        assert graph.topology.raw == ()
        assert graph.observations.raw[0].state_before == graph.initial[1].digest
    broken = copy.deepcopy(archive)
    broken["topology"]["parents"][0] = [0]
    with pytest.raises(ValueError, match="differs from complete replay"):
        priors.recipe_from_dict(broken)


def test_structural_geometry_charges_topology_calls_and_probability_parameters() -> None:
    request = _request("attachment")
    # Component size four, attachment fan-in two: 1 + 2 + 2 calls;
    # categorical supports 1 + (2+1) + (3+2) = 9 retained probability values.
    topology_calls, topology_numbers, probability_values = 5, 5, 9
    observation_calls = 4
    observation_numbers = request.topology.nodes * 2 + 3
    remaining = request.mechanism_raw_limits()
    assert remaining.draws == request.raw_limits.draws - topology_calls - observation_calls
    assert remaining.numbers == request.raw_limits.numbers - topology_numbers - observation_numbers
    expected_storage = 96 * (topology_numbers + observation_numbers) + 4096 * (
        topology_calls + observation_calls
    )
    expected_storage += 128 * probability_values + 4 * request.raw_limits.state_bytes
    assert remaining.working_bytes == request.raw_limits.working_bytes - expected_storage
    result = _sample(request)
    failing = replace(request, raw_limits=priors.RawDrawLimits(draws=len(result.graph.raw)))
    streams = [snapshot.restore() for snapshot in result.initial]
    with pytest.raises(ValueError, match="raw draw.*limit"):
        priors.sample_geometry_graph(
            failing,
            named_rng=streams[0],
            raw_rng=streams[1],
            calibration_rng=streams[2],
            fit_rng=streams[3],
        )
    for rng, snapshot in zip(streams, result.initial, strict=True):
        np.testing.assert_array_equal(rng.random(8), snapshot.restore().random(8))


@pytest.mark.parametrize("family", ["attachment", "motif-rewrites"])
@pytest.mark.parametrize(("task_type", "classes"), [("binary", 2), ("multiclass", 3)])
def test_structural_geometry_classification_adapters_preserve_topology(
    family: str, task_type: str, classes: int
) -> None:
    request = _request(family)
    request = replace(
        request, task=replace(request.task, task_type=cast(TaskType, task_type), classes=classes)
    )
    graph = _sample(request)
    assert priors.recipe_from_dict(json.loads(json.dumps(asdict(graph)))) == graph
    emitted = priors.generate_task(graph, seed=19, task_id=7)
    assert emitted.task_type == task_type
    assert priors.explain_generation(emitted, [1, 0, 1])["rows"] == [1, 0, 1]
    changed = priors.paired_recipe(graph, intervention="separation", value=0.25)
    assert isinstance(changed, priors.SampledGeometryGraph)
    assert changed.topology == graph.topology
    assert changed.graph.raw == graph.graph.raw


@pytest.mark.parametrize("engine", ["PCG64", "PCG64DXSM", "Philox", "SFC64", "MT19937"])
@pytest.mark.parametrize("family", ["attachment", "chain"])
def test_structural_geometry_reopens_advanced_engine_states(engine: str, family: str) -> None:
    streams = [
        np.random.Generator(getattr(np.random, engine)(seed)) for seed in (114, 115, 116, 117)
    ]
    for rng in streams:
        rng.random(3)
    graph = priors.sample_geometry_graph(
        _request(family),
        named_rng=streams[0],
        raw_rng=streams[1],
        calibration_rng=streams[2],
        fit_rng=streams[3],
    )
    assert graph.replay() == graph
    assert priors.recipe_from_dict(json.loads(json.dumps(asdict(graph)))) == graph
    for rng, final in zip(streams, graph.final, strict=True):
        np.testing.assert_array_equal(rng.random(8), final.restore().random(8))


def test_structural_geometry_strict_dispatch_caller_limits_and_empty_prefix_tampering() -> None:
    request = replace(_request("isolated-motif"), limits=priors.CausalLimits(parameters=1_000_001))
    graph = _sample(request)
    archive = json.loads(json.dumps(asdict(graph)))
    assert priors.recipe_from_dict(archive, limits=request.limits) == graph
    with pytest.raises(ValueError, match="declared limits exceed caller limits"):
        priors.recipe_from_dict(archive)
    emitted = priors.generate_task(graph, seed=19, task_id=7)
    assert priors.explain_generation(emitted, [0], limits=request.limits)["rows"] == [0]
    with pytest.raises(ValueError, match="declared limits exceed caller limits"):
        priors.explain_generation(emitted, [0])
    pair = priors.generate_pair(
        graph, intervention="noise", value=0.25, seed=19, task_id=7, limits=request.limits
    )
    assert all(json.loads(pair.card_json)["required_equalities"].values())
    changed_observations = replace(
        graph.observations,
        raw=(
            replace(graph.observations.raw[0], state_before="0" * 64),
            *graph.observations.raw[1:],
        ),
    )
    broken = replace(
        graph,
        observations=changed_observations,
        layout=replace(graph.layout, observations=changed_observations),
    )
    with pytest.raises(ValueError, match="raw stream boundaries differ"):
        broken.validate_structure()
    for version in ("unknown-topology-v1", None):
        changed = copy.deepcopy(archive)
        changed["request"]["topology"]["version"] = version
        with pytest.raises(ValueError, match="unsupported geometry topology prior version"):
            priors.recipe_from_dict(changed, limits=request.limits)
    changed = copy.deepcopy(archive)
    changed["request"]["topology"]["nodes"] = 3
    with pytest.raises(ValueError, match="unknown fields"):
        priors.recipe_from_dict(changed, limits=request.limits)


@pytest.mark.parametrize(
    ("nodes", "components", "edges"), [(2, 1, 1), (4, 1, 2), (7, 2, 2), (8, 3, 8), (5, 5, 1)]
)
def test_attachment_reservation_matches_retained_supports_across_component_sizes(
    nodes: int, components: int, edges: int
) -> None:
    prior = priors.AttachmentGraphPrior(nodes=nodes, components=components, edges_per_node=edges)
    request = replace(_request("attachment"), topology=prior)
    topology = priors.sample_structural_graph(prior, rng=np.random.default_rng(115))
    probabilities = sum(len(dict(record.parameters)["probabilities"]) for record in topology.raw)
    calls = len(topology.raw) + 4
    numbers = len(topology.raw) + 2 * nodes + 3
    remaining = request.mechanism_raw_limits()
    assert request.raw_limits.draws - remaining.draws == calls
    assert request.raw_limits.numbers - remaining.numbers == numbers
    storage = 96 * numbers + 4096 * calls + 128 * probabilities + 4 * request.raw_limits.state_bytes
    assert request.raw_limits.working_bytes - remaining.working_bytes == storage


def test_motif_reservation_charges_noop_upper_bound_and_rejects_prefix_only_budgets() -> None:
    prior = priors.MotifGraphPrior(
        depth=2, motifs=("chain",), rewrite_steps=3, rewrite_kinds=("path",)
    )
    request = replace(_request("chain"), topology=prior)
    topology = priors.sample_structural_graph(prior, rng=np.random.default_rng(115))
    assert all(receipt.reason == "no-admissible-replacement" for receipt in topology.rewrites)
    # Four motif substitutions, then at most three raw calls per rewrite even
    # though chain edges have no eligible intermediate node in this realization.
    bound = 4 + 3 * 3
    assert len(topology.raw) < bound
    remaining = request.mechanism_raw_limits()
    calls = bound + 4
    numbers = bound + 2 * prior.nodes + 3
    assert request.raw_limits.draws - remaining.draws == calls
    assert request.raw_limits.numbers - remaining.numbers == numbers
    storage = request.raw_limits.working_bytes - remaining.working_bytes
    for raw_limits in (
        priors.RawDrawLimits(draws=calls),
        priors.RawDrawLimits(numbers=numbers),
        priors.RawDrawLimits(working_bytes=storage),
    ):
        streams = [np.random.default_rng(seed) for seed in (114, 115, 116, 117)]
        untouched = copy.deepcopy(streams)
        with pytest.raises(ValueError, match="aggregate geometry raw draw limit"):
            priors.sample_geometry_graph(
                replace(request, raw_limits=raw_limits),
                named_rng=streams[0],
                raw_rng=streams[1],
                calibration_rng=streams[2],
                fit_rng=streams[3],
            )
        for rng, control in zip(streams, untouched, strict=True):
            np.testing.assert_array_equal(rng.random(8), control.random(8))
