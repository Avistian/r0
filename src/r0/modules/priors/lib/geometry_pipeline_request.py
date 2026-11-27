"""Opt-in nested request semantics for independently fitted numerical pipelines."""

from dataclasses import dataclass, replace

from r0.modules.priors.lib.causal_limits import CausalLimits
from r0.modules.priors.lib.geometry_graph_request import GeometryGraphRequest
from r0.modules.priors.lib.joint_pipeline_plans import JointPipelinePrior
from r0.modules.priors.lib.joint_width_plans import JointWidthPlan


@dataclass(frozen=True, slots=True)
class GeometryPipelineRequest(GeometryGraphRequest):
    pipeline_prior: JointPipelinePrior = JointPipelinePrior()
    version: str = "geometry-pipeline-request-v1"

    def __post_init__(self) -> None:
        GeometryGraphRequest.__post_init__(self)
        if (
            self.version != "geometry-pipeline-request-v1"
            or type(self.pipeline_prior) is not JointPipelinePrior
            or self.noise_scale != 0
        ):
            raise ValueError("sampled pipelines require their prior and no authored noise scale")

    def mechanism_limits(self, retained: tuple[object, ...]) -> CausalLimits:
        limits = GeometryGraphRequest.mechanism_limits(self, retained)
        widths = next((item for item in retained if type(item) is JointWidthPlan), None)
        if widths is None:
            raise ValueError("pipeline admission requires retained channel widths")
        # Four serialized/replay copies of fit inputs/constants plus stage buffers,
        # plan/owner records and the live complete ancestral sample. These are
        # conservative accounting bounds, not measured RSS. Raw draws share the
        # remaining graph ledger; no per-stage raw allowance is reset.
        numbers = 8 * (self.calibration_rows + 64) * sum(widths.widths)
        storage = 128 * numbers + 16384 * len(widths.widths)
        if numbers >= limits.parameters or storage >= limits.working_bytes:
            raise ValueError("aggregate sampled pipeline construction limit exceeded")
        return replace(
            limits,
            parameters=limits.parameters - numbers,
            working_bytes=limits.working_bytes - storage,
        )
