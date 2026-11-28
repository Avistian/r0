"""One retained shared scope for sampled geometry, parent functions and node pipelines."""

from dataclasses import dataclass

from r0.modules.priors.lib.geometry_parent_request import GeometryParentRequest
from r0.modules.priors.lib.joint_graph_request import JointGraphRequest
from r0.modules.priors.lib.materialized_parent_graph import MaterializedParentGraph
from r0.modules.priors.lib.sampled_geometry_graph import SampledGeometryGraph
from r0.modules.priors.lib.sampled_pipeline_geometry import (
    SampledPipelineGeometry,
    pipeline_operations,
)


@dataclass(frozen=True, slots=True)
class SampledParentGeometry(SampledPipelineGeometry):
    def extra_operations(self) -> tuple[tuple[str, ...], ...]:
        if (
            type(self.request) is not GeometryParentRequest
            or type(self.graph) is not MaterializedParentGraph
            or self.graph.composition != self.request.composition
        ):
            raise ValueError("parent geometry composition differs from request")
        if self.request.pipeline_prior is None:
            if self.pipelines:
                raise ValueError("parent geometry retains unrequested sampled pipelines")
            return ()
        return pipeline_operations(self, self.pipelines, self.request.pipeline_prior)

    def expected_graph_request(self) -> JointGraphRequest:
        if not self.pipelines:
            return SampledGeometryGraph.expected_graph_request(self)
        return SampledPipelineGeometry.expected_graph_request(self)
