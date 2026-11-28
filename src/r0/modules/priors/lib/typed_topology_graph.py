"""Atomic five-stream topology, geometry and typed mechanism envelope."""

from dataclasses import asdict, dataclass

import numpy as np

from r0.modules.priors.lib.converted_graph_request import ConvertedGraphRequest
from r0.modules.priors.lib.converted_graph_sampling import sample_converted_graph
from r0.modules.priors.lib.generator_snapshot import GeneratorSnapshot, snapshot_generator
from r0.modules.priors.lib.geometry_graph_request import GeometryGraphRequest
from r0.modules.priors.lib.geometry_graph_sampling import sample_geometry_graph
from r0.modules.priors.lib.native_graph_request import NativeGraphRequest
from r0.modules.priors.lib.native_graph_sampling import sample_native_graph
from r0.modules.priors.lib.native_vector_request import NativeVectorGraphRequest
from r0.modules.priors.lib.native_vector_sampling import sample_native_vector_graph
from r0.modules.priors.lib.raw_draw_capture import (
    raw_stream_states,
    raw_transactions,
    validate_raw_streams,
)
from r0.modules.priors.lib.sampled_converted_graph import SampledConvertedGraph
from r0.modules.priors.lib.sampled_geometry_graph import SampledGeometryGraph
from r0.modules.priors.lib.sampled_native_graph import SampledNativeGraph
from r0.modules.priors.lib.sampled_native_vector_graph import SampledNativeVectorGraph
from r0.modules.priors.lib.sampled_topology_hyperprior import (
    SampledTopologyHyperprior,
    sample_topology_hyperprior,
)
from r0.modules.priors.lib.typed_topology_adapters import typed_topology_child_request
from r0.modules.priors.lib.typed_topology_geometry import (
    NativeTopologyGeometry,
    sample_native_topology_geometry,
)
from r0.modules.priors.lib.typed_topology_request import TypedTopologyRequest

type TypedTopologyMechanisms = (
    SampledGeometryGraph | SampledNativeGraph | SampledNativeVectorGraph | SampledConvertedGraph
)


@dataclass(frozen=True, slots=True)
class TypedTopologyGraph:
    request: TypedTopologyRequest
    topology: SampledTopologyHyperprior
    geometry: NativeTopologyGeometry | None
    graph: TypedTopologyMechanisms
    initial: tuple[GeneratorSnapshot, ...]
    final: tuple[GeneratorSnapshot, ...]
    version: str = "sampled-typed-topology-r0-v1"

    @property
    def runtime(self):
        return (
            self.graph.runtime
            if isinstance(self.graph, SampledGeometryGraph)
            else self.graph.recipe
        )

    @property
    def task(self):
        return self.runtime.task

    @property
    def limits(self):
        return self.request.limits

    def validate(self) -> None:
        from r0.modules.priors.lib.typed_topology_decoding import typed_topology_graph_from_dict

        typed_topology_graph_from_dict(asdict(self), limits=self.limits)


def sample_typed_topology_graph(
    request: TypedTopologyRequest,
    *,
    topology_rng: np.random.Generator,
    named_rng: np.random.Generator,
    raw_rng: np.random.Generator,
    calibration_rng: np.random.Generator,
    fit_rng: np.random.Generator,
) -> TypedTopologyGraph:
    if type(request) is not TypedTopologyRequest:
        raise ValueError("typed topology requires its frozen request")
    request.admit()
    callers = dict(
        topology=topology_rng,
        named=named_rng,
        raw=raw_rng,
        calibration=calibration_rng,
        fit=fit_rng,
    )
    validate_raw_streams(callers)
    raw_stream_states(callers, request.raw_limits)
    with raw_transactions(callers, None) as (streams, _):
        initial = tuple(
            snapshot_generator(r, byte_limit=request.raw_limits.state_bytes)
            for r in streams.values()
        )
        topology = sample_topology_hyperprior(
            request.topology,
            rng=streams["topology"],
            limits=request.limits,
            raw_limits=request.raw_limits,
        )
        geometry = (
            sample_native_topology_geometry(request, len(topology.parents), streams["raw"])
            if request.native is not None
            else None
        )
        child = typed_topology_child_request(request, topology.parents, geometry)
        kwargs = dict(
            named_rng=streams["named"],
            raw_rng=streams["raw"],
            calibration_rng=streams["calibration"],
            fit_rng=streams["fit"],
        )
        if isinstance(child, NativeVectorGraphRequest):
            graph = sample_native_vector_graph(child, **kwargs)
        elif isinstance(child, NativeGraphRequest):
            graph = sample_native_graph(child, **kwargs)
        elif isinstance(child, ConvertedGraphRequest):
            graph = sample_converted_graph(child, **kwargs)
        else:
            assert isinstance(child, GeometryGraphRequest)
            graph = sample_geometry_graph(child, **kwargs)
        final = tuple(
            snapshot_generator(r, byte_limit=request.raw_limits.state_bytes)
            for r in streams.values()
        )
        return TypedTopologyGraph(request, topology, geometry, graph, initial, final)
