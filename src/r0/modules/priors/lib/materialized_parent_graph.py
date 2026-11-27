"""Parent mechanisms in a larger shared geometry/pipeline sampling scope."""

from dataclasses import dataclass

from r0.modules.priors.lib.functional_causal import FunctionalCausalRecipe
from r0.modules.priors.lib.generator_snapshot import GeneratorSnapshot
from r0.modules.priors.lib.joint_graph_admission import JointGraphAdmission, admit_joint_graph
from r0.modules.priors.lib.joint_graph_plans import JointGraphPlan
from r0.modules.priors.lib.joint_graph_request import JointGraphRequest
from r0.modules.priors.lib.joint_operation_registry import JointOperationRegistry
from r0.modules.priors.lib.joint_plan_ownership import JointPlanOwner, plan_owners
from r0.modules.priors.lib.joint_root_plans import JointRootPlan
from r0.modules.priors.lib.joint_root_realization import JointRootRealization
from r0.modules.priors.lib.parent_composition_plans import (
    ParentCompositionPrior,
    ParentFunctionPlan,
)
from r0.modules.priors.lib.parent_function_realization import (
    ParentCalibrationReceipt,
    ParentFunctionRealization,
)
from r0.modules.priors.lib.parent_graph_admission import admit_parent_graph
from r0.modules.priors.lib.raw_draw_records import RawDrawRecord
from r0.modules.priors.lib.sampled_functional_graph import SampledFunctionalGraph
from r0.modules.priors.lib.sampled_parent_graph import SampledParentGraph


@dataclass(frozen=True, slots=True)
class MaterializedParentGraph:
    request: JointGraphRequest
    plan: JointGraphPlan
    registry: JointOperationRegistry
    owners: tuple[JointPlanOwner, ...]
    realizations: tuple[JointRootRealization | ParentFunctionRealization, ...]
    calibrations: tuple[ParentCalibrationReceipt, ...]
    raw: tuple[RawDrawRecord, ...]
    admission: JointGraphAdmission
    runtime: FunctionalCausalRecipe
    composition: ParentCompositionPrior
    version: str = "materialized-parent-graph-v1"

    def explicit_graph(
        self, initial: tuple[GeneratorSnapshot, ...], final: tuple[GeneratorSnapshot, ...]
    ) -> SampledFunctionalGraph:
        raise ValueError("parent geometry scope cannot be serialized as a legacy explicit graph")

    def validate_structure(
        self, initial: tuple[GeneratorSnapshot, ...], final: tuple[GeneratorSnapshot, ...]
    ) -> None:
        if self.version != "materialized-parent-graph-v1":
            raise ValueError("unsupported materialized parent graph version")
        self.plan.validate(self.registry, owners=self.owners)
        if admit_joint_graph(self.request, self.plan, self.registry) != self.admission:
            raise ValueError("materialized parent admission differs from its complete graph plan")
        mechanisms = tuple(n.mechanism for n in self.plan.nodes)
        if any(not isinstance(p, JointRootPlan | ParentFunctionPlan) for p in mechanisms):
            raise ValueError("materialized parent graph contains legacy joint mechanism plans")
        plans = tuple(p for p in mechanisms if isinstance(p, JointRootPlan | ParentFunctionPlan))
        # An un-serialized adapter reuses the explicit parent validator. References
        # are translated from this envelope's nodes to the adapter's plan tuple;
        # the real typed owners above are validated against the retained graph plan.
        owners = [JointPlanOwner(("graph",), "parent-graph", ())]
        for i, p in enumerate(plans):
            owners.append(JointPlanOwner(("graph", f"node:{i}"), "node", ("plans", i)))
            owners.extend(
                p.owners(("plans", i))
                if isinstance(p, ParentFunctionPlan)
                else plan_owners(p, ("plans", i))
            )
        adapted = SampledParentGraph(
            self.request,
            self.composition,
            plans,
            self.registry,
            tuple(owners),
            self.realizations,
            self.calibrations,
            self.raw,
            initial,
            final,
            admit_parent_graph(self.request, plans, self.registry),
            self.runtime,
        )
        adapted.validate_structure()
