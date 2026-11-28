"""Atomic topology-first construction using the same four explicit caller streams."""

import numpy as np

from r0.modules.priors.lib.cauchy_graph import sample_cauchy_graph
from r0.modules.priors.lib.generator_snapshot import snapshot_generator
from r0.modules.priors.lib.joint_graph_sampling import sample_joint_graph
from r0.modules.priors.lib.raw_draw_capture import raw_transactions
from r0.modules.priors.lib.sampled_topology_graph import SampledTopologyGraph
from r0.modules.priors.lib.topology_graph_request import TopologyGraphRequest


def sample_topology_graph(
    request: TopologyGraphRequest,
    *,
    named_rng: np.random.Generator,
    raw_rng: np.random.Generator,
    calibration_rng: np.random.Generator,
    fit_rng: np.random.Generator,
) -> SampledTopologyGraph:
    if type(request) is not TopologyGraphRequest:
        raise ValueError("topology graph sampling requires a frozen request")
    child_limits = request.mechanism_limits()
    raw_limits = request.mechanism_raw_limits()
    callers = {"named": named_rng, "raw": raw_rng, "calibration": calibration_rng, "fit": fit_rng}
    with raw_transactions(callers, None) as (streams, _):
        initial = tuple(
            snapshot_generator(rng, byte_limit=request.raw_limits.state_bytes)
            for rng in streams.values()
        )
        topology = sample_cauchy_graph(request.topology, rng=streams["raw"], limits=request.limits)
        graph = sample_joint_graph(
            request.for_parents(topology.parents, limits=child_limits, raw_limits=raw_limits),
            named_rng=streams["named"],
            raw_rng=streams["raw"],
            calibration_rng=streams["calibration"],
            fit_rng=streams["fit"],
        )
        final = tuple(
            snapshot_generator(rng, byte_limit=request.raw_limits.state_bytes)
            for rng in streams.values()
        )
        result = SampledTopologyGraph(request, topology, graph, initial, final)
        result.validate_structure()
        return result
