"""Opt-in sampled geometry with source-specific parent composition and pipelines."""

from dataclasses import dataclass, replace

from r0.modules.priors.lib.causal_limits import CausalLimits
from r0.modules.priors.lib.geometry_graph_request import GeometryGraphRequest
from r0.modules.priors.lib.joint_pipeline_plans import JointPipelinePrior
from r0.modules.priors.lib.joint_width_plans import JointWidthPlan
from r0.modules.priors.lib.parent_composition_plans import ParentCompositionPrior


@dataclass(frozen=True, slots=True)
class GeometryParentRequest(GeometryGraphRequest):
    composition: ParentCompositionPrior = ParentCompositionPrior()
    pipeline_prior: JointPipelinePrior | None = None
    version: str = "geometry-parent-request-v1"

    def __post_init__(self) -> None:
        GeometryGraphRequest.__post_init__(self)
        if (
            type(self.composition) is not ParentCompositionPrior
            or self.pipeline_prior is not None
            and (type(self.pipeline_prior) is not JointPipelinePrior or self.noise_scale != 0)
        ):
            raise ValueError("parent geometry requires its composition and optional pipeline prior")

    def mechanism_limits(self, retained: tuple[object, ...]) -> CausalLimits:
        limits = GeometryGraphRequest.mechanism_limits(self, retained)
        if self.pipeline_prior is None:
            return limits
        widths = next((r for r in retained if type(r) is JointWidthPlan), None)
        if widths is None:
            raise ValueError("parent pipeline admission requires retained widths")
        numbers = 8 * (self.calibration_rows + 64) * sum(widths.widths)
        storage = 128 * numbers + 16384 * len(widths.widths)
        if numbers >= limits.parameters or storage >= limits.working_bytes:
            raise ValueError("aggregate parent pipeline construction limit exceeded")
        return replace(
            limits,
            parameters=limits.parameters - numbers,
            working_bytes=limits.working_bytes - storage,
        )
