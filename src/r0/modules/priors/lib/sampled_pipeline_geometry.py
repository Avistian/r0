"""Versioned pipeline child records inside the stable sampled geometry envelope."""

from dataclasses import dataclass, replace

from r0.modules.priors.lib.geometry_pipeline_request import GeometryPipelineRequest
from r0.modules.priors.lib.joint_graph_request import JointGraphNoise, JointGraphRequest
from r0.modules.priors.lib.joint_pipeline_plans import JointPipelinePrior
from r0.modules.priors.lib.joint_pipeline_realization import JointPipelineRealization
from r0.modules.priors.lib.sampled_geometry_graph import SampledGeometryGraph


@dataclass(frozen=True, slots=True)
class SampledPipelineGeometry(SampledGeometryGraph):
    pipelines: tuple[JointPipelineRealization, ...] = ()

    def extra_operations(self) -> tuple[tuple[str, ...], ...]:
        if type(self.request) is not GeometryPipelineRequest:
            raise ValueError("sampled pipeline envelope requires its opt-in request")
        return pipeline_operations(self, self.pipelines, self.request.pipeline_prior)

    def expected_graph_request(self) -> JointGraphRequest:
        original = SampledGeometryGraph.expected_graph_request(self)
        return replace(
            original,
            authored_pipelines=tuple(p.pipeline for p in self.pipelines),
            noise=tuple(JointGraphNoise((p.plan.sigma,) * p.plan.width) for p in self.pipelines),
        )


def pipeline_operations(
    self: SampledGeometryGraph,
    pipelines: tuple[JointPipelineRealization, ...],
    prior: JointPipelinePrior,
) -> tuple[tuple[str, ...], ...]:
    if (
        type(pipelines) is not tuple
        or len(pipelines) != len(self.graph.plan.nodes)
        or any(type(p) is not JointPipelineRealization for p in pipelines)
    ):
        raise ValueError("sampled pipeline envelope requires one realization per node")
    result: list[tuple[str, ...]] = []
    previous_stop = 0
    for i, fitted in enumerate(pipelines):
        planned = fitted.plan
        planned.validate(self.graph.registry.decisions)
        if (
            planned.operation != (f"pipeline:{i}",)
            or planned.width != self.widths.widths[i]
            or planned.prior != prior
            or fitted.version != "independent-node-pipeline-realization-v1"
            or fitted.pipeline != self.runtime.nodes[i].pipeline
            or fitted.pipeline.node_importance != planned.importance
            or fitted.calibration.identity != f"pipeline-node-{i}-fit"
            or len(fitted.calibration.sample) != self.request.calibration_rows
            or any(len(row) != planned.width for row in fitted.calibration.sample)
            or type(fitted.raw_start) is not int
            or type(fitted.raw_stop) is not int
            or not previous_stop <= fitted.raw_start < fitted.raw_stop <= len(self.graph.raw)
        ):
            raise ValueError("sampled pipeline geometry or fit binding differs")
        groups = [""] * self.layout.converter_channels[i]
        for group, spans in zip(self.request.groups, self.layout.spans, strict=True):
            for node, start, stop in spans:
                if node == i:
                    groups[start:stop] = [group.name] * (stop - start)
        if tuple(c.feature_group for c in planned.converters) != tuple(groups):
            raise ValueError("pipeline converter group binding differs")
        if any(
            r.owner[:1] != planned.operation
            for r in self.graph.raw[fitted.raw_start : fitted.raw_stop]
        ):
            raise ValueError("pipeline raw receipt has foreign owners")
        previous_stop = fitted.raw_stop
        result.extend(owner.operation for owner in planned.owners())
    return tuple(result)
