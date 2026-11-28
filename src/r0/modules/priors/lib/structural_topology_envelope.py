"""Retain source topology receipts and bind them to the dedicated topology stream."""

from dataclasses import dataclass
from typing import Any, cast

import numpy as np

from r0.modules.priors.lib.cauchy_graph import CauchyGraph, CauchyGraphPrior
from r0.modules.priors.lib.causal_limits import CausalLimits
from r0.modules.priors.lib.fixed_parent_topology import FixedParentTopology, RetainedParentTopology
from r0.modules.priors.lib.generator_snapshot import GeneratorSnapshot, snapshot_generator
from r0.modules.priors.lib.geometry_topology import (
    decode_geometry_topology,
    sample_geometry_topology,
)
from r0.modules.priors.lib.raw_draw_records import RawDrawLimits, RawDrawRecord
from r0.modules.priors.lib.structural_graph import StructuralGraph
from r0.modules.priors.lib.structural_graph_algorithm import GraphRewrite
from r0.modules.priors.lib.structural_topology_request import StructuralTopologyRequest
from r0.modules.priors.lib.topology_hyperprior_decoding import complete_topology_record as record


@dataclass(frozen=True, slots=True)
class SampledStructuralTopology:
    request: StructuralTopologyRequest
    graph: CauchyGraph | StructuralGraph
    initial: GeneratorSnapshot
    final: GeneratorSnapshot
    limits: CausalLimits
    raw_limits: RawDrawLimits
    version: str = "sampled-structural-topology-r0-v1"

    @property
    def parents(self):
        return self.graph.parents

    @property
    def raw(self):
        return self.graph.raw

    @property
    def components(self):
        return RetainedParentTopology(FixedParentTopology(self.parents), self.limits).components


def sample_structural_topology(
    request: StructuralTopologyRequest,
    *,
    rng: np.random.Generator,
    limits: CausalLimits,
    raw_limits: RawDrawLimits,
) -> SampledStructuralTopology:
    request.admit(limits, raw_limits)
    initial = snapshot_generator(rng, byte_limit=raw_limits.state_bytes)
    graph = sample_geometry_topology(request.prior, rng=rng, limits=limits, raw_limits=raw_limits)
    assert isinstance(graph, (CauchyGraph, StructuralGraph))
    return SampledStructuralTopology(
        request,
        graph,
        initial,
        snapshot_generator(rng, byte_limit=raw_limits.state_bytes),
        limits,
        raw_limits,
    )


def structural_topology_request_from_dict(data: Any) -> StructuralTopologyRequest:
    raw = record(data, StructuralTopologyRequest)
    raw["prior"] = decode_geometry_topology(raw["prior"])
    return StructuralTopologyRequest(**raw)


def structural_topology_metadata(
    data: Any,
    *,
    request: StructuralTopologyRequest,
    limits: CausalLimits,
    raw_limits: RawDrawLimits,
):
    request.admit(limits, raw_limits)
    raw = record(data, SampledStructuralTopology)
    if (
        raw["version"] != "sampled-structural-topology-r0-v1"
        or structural_topology_request_from_dict(raw["request"]) != request
        or CausalLimits(**record(raw["limits"], CausalLimits)) != limits
        or RawDrawLimits(**record(raw["raw_limits"], RawDrawLimits)) != raw_limits
    ):
        raise ValueError("structural topology receipt differs from original request")
    kind = CauchyGraph if isinstance(request.prior, CauchyGraphPrior) else StructuralGraph
    graph = record(raw["graph"], kind)
    if (
        decode_geometry_topology(graph["prior"]) != request.prior
        or CausalLimits(**record(graph["limits"], CausalLimits)) != limits
    ):
        raise ValueError("structural topology source graph differs from original request")
    graph["prior"], graph["limits"] = request.prior, limits
    graph["parents"] = tuple(tuple(row) for row in graph["parents"])
    draws: list[RawDrawRecord] = []
    for item in graph["raw"]:
        draw = record(item, RawDrawRecord)
        for key in ("owner", "shape", "values"):
            draw[key] = tuple(draw[key])
        draw["parameters"] = tuple(tuple(pair) for pair in draw["parameters"])
        # Categorical probability vectors are nested raw parameters.
        draw["parameters"] = tuple(
            (key, tuple(cast(list[Any], value)) if isinstance(value, list) else value)
            for key, value in draw["parameters"]
        )
        draws.append(RawDrawRecord(**draw))
    graph["raw"] = tuple(draws)
    if kind is StructuralGraph:
        rewrites: list[GraphRewrite] = []
        for item in graph["rewrites"]:
            change = record(item, GraphRewrite)
            change["subject"] = tuple(change["subject"])
            for key in ("removed", "added"):
                change[key] = tuple(tuple(edge) for edge in change[key])
            rewrites.append(GraphRewrite(**change))
        graph["rewrites"] = tuple(rewrites)
    source = kind(**graph)
    source.validate()
    initial = GeneratorSnapshot(**record(raw["initial"], GeneratorSnapshot))
    final = GeneratorSnapshot(**record(raw["final"], GeneratorSnapshot))
    initial.validate_metadata(byte_limit=raw_limits.state_bytes)
    final.validate_metadata(byte_limit=raw_limits.state_bytes)
    if (
        initial.engine != final.engine
        or (
            draws
            and (draws[0].state_before != initial.digest or draws[-1].state_after != final.digest)
        )
        or (not draws and initial != final)
    ):
        raise ValueError("structural topology source stream differs from its envelope")
    return SampledStructuralTopology(request, source, initial, final, limits, raw_limits)
