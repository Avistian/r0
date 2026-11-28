"""Bounded reopening of topology and mechanism archives as one construction."""

from dataclasses import fields

from r0.modules.priors.lib.cauchy_graph import CauchyGraphPrior
from r0.modules.priors.lib.causal_limits import CausalLimits
from r0.modules.priors.lib.function_metadata import admit_function_metadata
from r0.modules.priors.lib.functional_causal_decoding import functional_array, functional_record
from r0.modules.priors.lib.generator_snapshot import GeneratorSnapshot
from r0.modules.priors.lib.sampled_graph_decoding import (
    construction_matches,
    decode_joint_graph_request,
)
from r0.modules.priors.lib.sampled_topology_graph import SampledTopologyGraph
from r0.modules.priors.lib.topology_graph_request import TopologyGraphRequest
from r0.modules.priors.lib.topology_graph_sampling import sample_topology_graph

_DEFAULT_LIMITS = CausalLimits()


def topology_graph_from_dict(
    data: object, *, limits: CausalLimits = _DEFAULT_LIMITS
) -> SampledTopologyGraph:
    """Replay all four initial streams and reject every changed retained field."""
    if type(limits) is not CausalLimits:
        raise ValueError("topology graph decoding requires caller-owned limits")
    admit_function_metadata(data, limits)
    try:
        record = functional_record(data, SampledTopologyGraph)
        if record.get("version") != "sampled-topology-graph-v1":
            raise ValueError("unsupported sampled topology graph version")
        raw = functional_record(record["request"], TopologyGraphRequest)
        if raw.pop("version") != "topology-graph-request-v1":
            raise ValueError("unsupported topology graph request version")
        topology = CauchyGraphPrior(**functional_record(raw.pop("topology"), CauchyGraphPrior))
        # Decode common controls using the same strict contract as explicit DAGs.
        # This geometry is only preflight; replay samples the actual parents.
        raw["parents"] = tuple(() for _ in functional_array(raw["widths"]))
        common = decode_joint_graph_request(raw, limits)
        request = TopologyGraphRequest(
            topology=topology,
            **{
                f.name: getattr(common, f.name)
                for f in fields(common)
                if f.name not in ("parents", "version")
            },
        )
        admit_function_metadata(data, request.limits)
        initial = functional_array(record["initial"])
        if len(initial) != 4:
            raise ValueError("topology graph requires four initial stream snapshots")
        snapshots = tuple(
            GeneratorSnapshot(**functional_record(s, GeneratorSnapshot)) for s in initial
        )
        streams = tuple(s.restore(byte_limit=request.raw_limits.state_bytes) for s in snapshots)
        graph = sample_topology_graph(
            request,
            named_rng=streams[0],
            raw_rng=streams[1],
            calibration_rng=streams[2],
            fit_rng=streams[3],
        )
        if not construction_matches(graph, data):
            raise ValueError("topology graph archive differs from complete replay")
        return graph
    except (TypeError, KeyError, OverflowError, RecursionError) as exc:
        raise ValueError("invalid topology graph archive") from exc
