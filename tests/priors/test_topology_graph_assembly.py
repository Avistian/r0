"""Sample topology and mechanisms as one replayable, atomic graph construction."""

import copy
import json
from dataclasses import asdict, replace

import numpy as np
import pytest

from r0.modules import priors


def test_topology_assembly_retains_topology_and_replays_all_caller_streams() -> None:
    functions = priors.JointFunctionPrior(
        functions="tree", matrix=priors.JointMatrixPrior(families=("gaussian",))
    )
    request = priors.TopologyGraphRequest(
        topology=priors.CauchyGraphPrior(nodes=3, offset=1000.0),
        task=priors.LinearRecipe(
            rows=16, context_rows=8, features=1, signal_features=1, active_features=1
        ),
        widths=(1, 2, 1),
        observed=((0, 0),),
        target=((2, 0),),
        calibration_rows=4,
        root_prior=priors.JointRootPrior(families=("gaussian",), function=functions),
        function_prior=functions,
    )
    streams = [np.random.default_rng(seed) for seed in (114, 115, 116, 117)]
    controls = copy.deepcopy(streams)
    result = priors.sample_topology_graph(
        request,
        named_rng=streams[0],
        raw_rng=streams[1],
        calibration_rng=streams[2],
        fit_rng=streams[3],
    )
    assert result.topology.parents == ((), (0,), (0, 1))
    assert tuple(n.parents for n in result.graph.runtime.nodes) == result.topology.parents
    assert result.graph.runtime.nodes[2].function.input_width == 3
    assert result.replay() == result
    from r0.modules.priors.lib.topology_graph_decoding import topology_graph_from_dict

    archive = json.loads(json.dumps(asdict(result)))
    retained = copy.deepcopy(archive)
    assert topology_graph_from_dict(archive) == result
    assert archive == retained
    assert priors.recipe_from_dict(retained) == result
    emitted = priors.generate_task(result, seed=19, task_id=7)
    saved = json.loads(emitted.recipe_json)
    assert priors.recipe_from_dict(saved["recipe"]) == result
    assert priors.explain_generation(emitted, [7, 0, 7])["rows"] == [7, 0, 7]
    resplit = result.with_task(replace(result.task, context_rows=4, split=1))
    other = priors.generate_task(resplit, seed=19, task_id=7)
    assert emitted.source_task_id == other.source_task_id
    assert emitted.episode_id != other.episode_id
    np.testing.assert_array_equal(emitted.inputs.numpy(), other.inputs.numpy())
    np.testing.assert_array_equal(emitted.targets.numpy(), other.targets.numpy())
    pair = priors.generate_pair(result, intervention="noise", value=0.25, seed=19, task_id=7)
    assert all(json.loads(pair.card_json)["required_equalities"].values())
    archive["topology"]["raw"][3]["values"][0] += 0.01
    with pytest.raises(ValueError, match="differs from complete replay"):
        topology_graph_from_dict(archive)
    assert retained == json.loads(json.dumps(asdict(result)))
    repeated = priors.sample_topology_graph(
        request,
        named_rng=controls[0],
        raw_rng=controls[1],
        calibration_rng=controls[2],
        fit_rng=controls[3],
    )
    assert result == repeated
    for rng, control in zip(streams, controls, strict=True):
        np.testing.assert_array_equal(rng.random(8), control.random(8))
    # The topology has already drawn when the downstream mechanism budget fails.
    controls = copy.deepcopy(streams)
    failing = replace(request, raw_limits=priors.RawDrawLimits(draws=5))
    with pytest.raises(ValueError, match="raw draw.*limit"):
        priors.sample_topology_graph(
            failing,
            named_rng=streams[0],
            raw_rng=streams[1],
            calibration_rng=streams[2],
            fit_rng=streams[3],
        )
    for rng, control in zip(streams, controls, strict=True):
        np.testing.assert_array_equal(rng.random(8), control.random(8))
    # A budget sufficient for mechanisms alone must include topology's four draws.
    limited = replace(request, raw_limits=priors.RawDrawLimits(draws=len(result.graph.raw)))
    fresh = [snapshot.restore() for snapshot in result.initial]
    with pytest.raises(ValueError, match="raw draw.*limit"):
        priors.sample_topology_graph(
            limited,
            named_rng=fresh[0],
            raw_rng=fresh[1],
            calibration_rng=fresh[2],
            fit_rng=fresh[3],
        )
    for rng, snapshot in zip(fresh, result.initial, strict=True):
        np.testing.assert_array_equal(rng.random(8), snapshot.restore().random(8))


@pytest.mark.parametrize("engine", ["PCG64", "PCG64DXSM", "Philox", "SFC64", "MT19937"])
def test_topology_archive_retains_engine_state_and_rejects_nested_changes(engine: str) -> None:
    functions = priors.JointFunctionPrior(
        functions="lin", matrix=priors.JointMatrixPrior(families=("gaussian",))
    )
    limits = priors.CausalLimits(parameters=1_000_001)
    request = priors.TopologyGraphRequest(
        topology=priors.CauchyGraphPrior(nodes=2, offset=0.0),
        task=priors.LinearRecipe(
            rows=16, context_rows=8, features=1, signal_features=1, active_features=1
        ),
        widths=(1, 1),
        observed=((0, 0),),
        target=((1, 0),),
        calibration_rows=4,
        root_prior=priors.JointRootPrior(families=("gaussian",), function=functions),
        function_prior=functions,
        limits=limits,
    )
    streams = [np.random.Generator(getattr(np.random, engine)(seed)) for seed in range(4)]
    for stream in streams:
        stream.random(3)
    graph = priors.sample_topology_graph(
        request,
        named_rng=streams[0],
        raw_rng=streams[1],
        calibration_rng=streams[2],
        fit_rng=streams[3],
    )
    raw = json.loads(json.dumps(asdict(graph)))
    assert priors.recipe_from_dict(raw, limits=limits) == graph
    with pytest.raises(ValueError, match="declared limits exceed caller limits"):
        priors.recipe_from_dict(raw)
    for field in ("topology", "graph", "initial", "final"):
        changed = copy.deepcopy(raw)
        del changed[field]
        with pytest.raises(
            ValueError, match="invalid topology graph archive|differs from complete replay"
        ):
            priors.recipe_from_dict(changed, limits=limits)
    changed = copy.deepcopy(raw)
    changed["graph"]["raw"][0]["role"] = "altered"
    with pytest.raises(ValueError, match="differs from complete replay"):
        priors.recipe_from_dict(changed, limits=limits)
    emitted = priors.generate_task(graph, seed=3, task_id=4)
    assert priors.explain_generation(emitted, [2, 0, 2], limits=limits)["rows"] == [2, 0, 2]
    with pytest.raises(ValueError, match="declared limits exceed caller limits"):
        priors.explain_generation(emitted, [0])
