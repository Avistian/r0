"""Complete geometry and mechanism provenance under one named preference scope."""

from dataclasses import dataclass, replace

from r0.modules.priors.lib.cauchy_graph import CauchyGraph
from r0.modules.priors.lib.functional_causal import FunctionalCausalRecipe
from r0.modules.priors.lib.functional_graph_admission import admit_functional_graph
from r0.modules.priors.lib.generator_snapshot import GeneratorSnapshot
from r0.modules.priors.lib.geometry_graph_request import GeometryGraphRequest
from r0.modules.priors.lib.geometry_pipeline_request import GeometryPipelineRequest
from r0.modules.priors.lib.geometry_topology import GeometryTopology
from r0.modules.priors.lib.joint_graph_request import JointGraphRequest
from r0.modules.priors.lib.joint_observation_realization import JointObservationRealization
from r0.modules.priors.lib.joint_width_plans import JointWidthPlan
from r0.modules.priors.lib.linear import LinearRecipe
from r0.modules.priors.lib.materialized_joint_graph import MaterializedJointGraph
from r0.modules.priors.lib.observation_channel_layout import ObservationChannelLayout
from r0.modules.priors.lib.structural_graph import StructuralGraph


@dataclass(frozen=True, slots=True)
class SampledGeometryGraph:
    request: GeometryGraphRequest
    topology: GeometryTopology
    observations: JointObservationRealization
    layout: ObservationChannelLayout
    widths: JointWidthPlan
    graph: MaterializedJointGraph
    initial: tuple[GeneratorSnapshot, ...]
    final: tuple[GeneratorSnapshot, ...]
    version: str = "sampled-geometry-graph-v1"

    @property
    def runtime(self) -> FunctionalCausalRecipe:
        return self.graph.runtime

    @property
    def task(self) -> LinearRecipe:
        return self.request.task

    def geometry_records(self) -> tuple[object, ...]:
        return (self.topology, self.observations, self.layout, self.widths, self.initial)

    def validate_structure(self) -> None:
        from r0.modules.priors.lib.sampled_pipeline_geometry import SampledPipelineGeometry

        if (
            type(self) not in (SampledGeometryGraph, SampledPipelineGeometry)
            or self.version != "sampled-geometry-graph-v1"
            or type(self.request) not in (GeometryGraphRequest, GeometryPipelineRequest)
            or type(self.topology) not in (CauchyGraph, StructuralGraph)
            or type(self.observations) is not JointObservationRealization
            or type(self.layout) is not ObservationChannelLayout
            or type(self.widths) is not JointWidthPlan
            or type(self.graph) is not MaterializedJointGraph
        ):
            raise ValueError("sampled geometry requires complete frozen versioned records")
        if (
            type(self.initial) is not tuple
            or type(self.final) is not tuple
            or len(self.initial) != 4
            or len(self.final) != 4
            or any(type(s) is not GeneratorSnapshot for s in (*self.initial, *self.final))
        ):
            raise ValueError("sampled geometry requires four initial/final stream snapshots")
        admit_functional_graph(
            self.runtime.nodes,
            self.request.limits,
            rows=max(self.task.rows, self.request.calibration_rows),
            metadata=(self,),
            features=self.task.features,
            outputs=len(self.graph.request.target),
        )
        self.topology.validate()
        registry = self.graph.registry
        self.observations.validate(registry.decisions)
        self.layout.validate(registry.decisions)
        self.widths.validate(registry.decisions)
        expected_operations = (self.observations.plan.operation, self.widths.operation)
        expected_operations += tuple(owner.operation for owner in self.graph.plan.owners())
        expected_operations += self.extra_operations()
        if tuple(span.operation for span in registry.operations) != expected_operations:
            raise ValueError("geometry construction registry has missing or foreign owners")
        if (
            self.topology.prior != self.request.topology
            or self.topology.limits != self.request.limits
            or self.observations.plan.nodes != self.request.topology.nodes
            or self.observations.plan.groups != self.request.groups
            or self.observations.plan.subsample != self.request.subsample
            or self.observations.plan.limits != self.request.limits
            or self.observations.plan.operation != ("observations",)
            or self.widths.operation != ("widths",)
            or self.widths.prior != self.request.width_prior
            or self.widths.limits != self.request.limits
            or self.widths.converter_channels != self.layout.converter_channels
            or self.layout.observations != self.observations
            or self.layout.converter_widths != tuple((1,) * g.count for g in self.request.groups)
        ):
            raise ValueError("geometry construction bindings differ from requested controls")
        if self.graph.request != self.expected_graph_request():
            raise ValueError("geometry mechanism request differs from retained geometry")
        self.graph.validate_structure(self.initial, self.final)
        roles = ("named", "raw", "calibration", "fit")
        states = {role: snapshot.digest for role, snapshot in zip(roles, self.initial, strict=True)}
        engines = {
            role: snapshot.engine for role, snapshot in zip(roles, self.initial, strict=True)
        }
        for record in (*self.topology.raw, *self.observations.raw, *self.graph.raw):
            if (
                record.stream not in ("raw", "calibration", "fit")
                or record.generator != engines[record.stream]
                or record.state_before != states[record.stream]
            ):
                raise ValueError("geometry construction raw stream boundaries differ")
            states[record.stream] = record.state_after
        for role, snapshot in zip(roles, self.final, strict=True):
            if snapshot.engine != engines[role] or (
                role != "named" and snapshot.digest != states[role]
            ):
                raise ValueError("geometry final stream boundary differs")
        # Named preferences are not raw-ledger draws. Their initial/final state
        # and every retained preference are checked by complete construction replay.

    def extra_operations(self) -> tuple[tuple[str, ...], ...]:
        if type(self.request) is not GeometryGraphRequest:
            raise ValueError("plain geometry envelope cannot contain pipeline request semantics")
        return ()

    def expected_graph_request(self) -> JointGraphRequest:
        child_limits = self.request.mechanism_limits(self.geometry_records())
        return self.request.graph_request(
            self.topology.parents, self.layout, self.widths, limits=child_limits
        )

    def validate(self) -> None:
        self.replay()

    def replay(self) -> "SampledGeometryGraph":
        self.validate_structure()
        rebuilt = self._reconstruct(self.request)
        if rebuilt != self:
            raise ValueError("sampled geometry graph differs from complete replay")
        return rebuilt

    def with_task(self, task: LinearRecipe) -> "SampledGeometryGraph":
        self.validate()
        return self._reconstruct(replace(self.request, task=task))

    def _reconstruct(self, request: GeometryGraphRequest) -> "SampledGeometryGraph":
        from r0.modules.priors.lib.geometry_graph_sampling import sample_geometry_graph

        streams = tuple(
            s.restore(byte_limit=self.request.raw_limits.state_bytes) for s in self.initial
        )
        return sample_geometry_graph(
            request,
            named_rng=streams[0],
            raw_rng=streams[1],
            calibration_rng=streams[2],
            fit_rng=streams[3],
        )
