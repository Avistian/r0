"""Mechanisms constructed from a retained shared registry, without a new sampler scope."""

from dataclasses import dataclass

from r0.modules.priors.lib.functional_causal import FunctionalCausalRecipe
from r0.modules.priors.lib.generator_snapshot import GeneratorSnapshot
from r0.modules.priors.lib.joint_graph_admission import JointGraphAdmission
from r0.modules.priors.lib.joint_graph_plans import JointGraphPlan
from r0.modules.priors.lib.joint_graph_request import JointGraphRequest
from r0.modules.priors.lib.joint_operation_registry import JointOperationRegistry
from r0.modules.priors.lib.joint_plan_ownership import JointPlanOwner
from r0.modules.priors.lib.joint_root_realization import (
    JointRootRealization,
    RootFunctionRealization,
)
from r0.modules.priors.lib.raw_draw_records import RawDrawRecord
from r0.modules.priors.lib.sampled_functional_graph import (
    GraphCalibrationReceipt,
    SampledFunctionalGraph,
)


@dataclass(frozen=True, slots=True)
class MaterializedJointGraph:
    request: JointGraphRequest
    plan: JointGraphPlan
    registry: JointOperationRegistry
    owners: tuple[JointPlanOwner, ...]
    realizations: tuple[JointRootRealization | RootFunctionRealization, ...]
    calibrations: tuple[GraphCalibrationReceipt, ...]
    raw: tuple[RawDrawRecord, ...]
    admission: JointGraphAdmission
    runtime: FunctionalCausalRecipe
    version: str = "materialized-joint-graph-v1"

    def explicit_graph(
        self, initial: tuple[GeneratorSnapshot, ...], final: tuple[GeneratorSnapshot, ...]
    ) -> SampledFunctionalGraph:
        """Create v1 provenance only when the complete scope is the explicit graph."""
        if any(span.operation[:1] != ("graph",) for span in self.registry.operations):
            raise ValueError("shared geometry scope cannot become explicit graph v1 provenance")
        return self._sampled(initial, final)

    def validate_structure(
        self, initial: tuple[GeneratorSnapshot, ...], final: tuple[GeneratorSnapshot, ...]
    ) -> None:
        self._sampled(initial, final).validate_structure()

    def _sampled(
        self, initial: tuple[GeneratorSnapshot, ...], final: tuple[GeneratorSnapshot, ...]
    ) -> SampledFunctionalGraph:
        """Internal adapter for the original explicit-DAG envelope and its structural checks.

        Complete replay belongs to the enclosing sampler, whose scope may also
        include geometry. This adapter must not be serialized as that envelope.
        """
        if self.version != "materialized-joint-graph-v1":
            raise ValueError("unsupported materialized graph version")
        return SampledFunctionalGraph(
            self.request,
            self.plan,
            self.registry,
            self.owners,
            self.realizations,
            self.calibrations,
            self.raw,
            initial,
            final,
            self.admission,
            self.runtime,
        )
