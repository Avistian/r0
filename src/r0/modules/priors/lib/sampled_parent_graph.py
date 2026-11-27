"""Complete replayable per-parent graph construction, with unchanged legacy archives."""

from dataclasses import dataclass, replace

from r0.modules.priors.lib.functional_causal import FunctionalCausalRecipe
from r0.modules.priors.lib.functional_graph_admission import admit_functional_graph
from r0.modules.priors.lib.generator_snapshot import GeneratorSnapshot
from r0.modules.priors.lib.joint_graph_admission import JointGraphAdmission
from r0.modules.priors.lib.joint_graph_request import JointGraphNoise, JointGraphRequest
from r0.modules.priors.lib.joint_operation_registry import JointOperationRegistry
from r0.modules.priors.lib.joint_plan_ownership import JointPlanOwner, plan_owners
from r0.modules.priors.lib.joint_root_plans import JointRootPlan
from r0.modules.priors.lib.joint_root_realization import JointRootRealization
from r0.modules.priors.lib.linear import LinearRecipe
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


@dataclass(frozen=True, slots=True)
class SampledParentGraph:
    request: JointGraphRequest
    composition: ParentCompositionPrior
    plans: tuple[JointRootPlan | ParentFunctionPlan, ...]
    registry: JointOperationRegistry
    owners: tuple[JointPlanOwner, ...]
    realizations: tuple[JointRootRealization | ParentFunctionRealization, ...]
    calibrations: tuple[ParentCalibrationReceipt, ...]
    raw: tuple[RawDrawRecord, ...]
    initial: tuple[GeneratorSnapshot, ...]
    final: tuple[GeneratorSnapshot, ...]
    admission: JointGraphAdmission
    runtime: FunctionalCausalRecipe
    version: str = "sampled-parent-graph-v1"

    @property
    def task(self) -> LinearRecipe:
        return self.request.task

    def with_task(self, task: LinearRecipe) -> "SampledParentGraph":
        self.validate()
        return self._reconstruct(replace(self.request, task=task))

    def _reconstruct(self, request: JointGraphRequest) -> "SampledParentGraph":
        from r0.modules.priors.lib.parent_graph_sampling import sample_parent_graph

        streams = tuple(s.restore(byte_limit=request.raw_limits.state_bytes) for s in self.initial)
        return sample_parent_graph(
            request,
            composition=self.composition,
            named_rng=streams[0],
            raw_rng=streams[1],
            calibration_rng=streams[2],
            fit_rng=streams[3],
        )

    def replay(self) -> "SampledParentGraph":
        self.validate_structure()
        result = self._reconstruct(self.request)
        if result != self:
            raise ValueError("parent graph replay differs from retained construction")
        return result

    def validate(self) -> None:
        self.replay()

    def validate_structure(self) -> None:
        if (
            self.version != "sampled-parent-graph-v1"
            or type(self.request) is not JointGraphRequest
            or type(self.composition) is not ParentCompositionPrior
            or type(self.runtime) is not FunctionalCausalRecipe
            or type(self.plans) is not tuple
            or len(self.plans) != len(self.request.parents)
            or len(self.realizations) != len(self.plans)
            or len(self.initial) != 4
            or len(self.final) != 4
            or any(type(s) is not GeneratorSnapshot for s in (*self.initial, *self.final))
        ):
            raise ValueError("parent graph requires its complete versioned frozen envelope")
        admit_functional_graph(
            self.runtime.nodes,
            self.request.limits,
            rows=max(self.task.rows, self.request.calibration_rows),
            metadata=(self,),
            features=self.task.features,
            outputs=len(self.request.target),
        )
        if admit_parent_graph(self.request, self.plans, self.registry) != self.admission:
            raise ValueError("parent graph admission differs")
        derived = [JointPlanOwner(("graph",), "parent-graph", ())]
        for i, (plan, realized, node) in enumerate(
            zip(self.plans, self.realizations, self.runtime.nodes, strict=True)
        ):
            if type(plan) not in (JointRootPlan, ParentFunctionPlan):
                raise ValueError("parent graph plan kind differs")
            if isinstance(plan, ParentFunctionPlan):
                if (
                    not self.request.parents[i]
                    or plan.prior != self.composition
                    or plan.parent_widths
                    != tuple(self.request.widths[p] for p in self.request.parents[i])
                    or type(realized) is not ParentFunctionRealization
                ):
                    raise ValueError("parent graph sliced plan or realization differs")
                function, root = realized.function, None
                member_realizations = (
                    *((realized.joint,) if realized.joint else ()),
                    *realized.children,
                    *((realized.reducer,) if realized.reducer else ()),
                )
                if tuple(r.plan for r in member_realizations) != plan.functions:
                    raise ValueError("parent realized member plans differ")
                realized.validate(self.registry.decisions)
            else:
                if self.request.parents[i] or type(realized) is not JointRootRealization:
                    raise ValueError("parent graph root plan or realization differs")
                realized.validate(self.registry.decisions)
                function, root = realized.function.function, realized.calibration.base
            if realized.plan != plan or plan.operation != ("graph", f"node:{i}", "mechanism"):
                raise ValueError("parent graph realization plan or ownership differs")
            noise = self.request.noise[i] if self.request.noise else JointGraphNoise()
            pipeline = (
                self.request.authored_pipelines[i] if self.request.authored_pipelines else None
            )
            if (
                node.function != function
                or node.root != root
                or node.function.output_width != self.request.widths[i]
                or node.noise_scale != noise.scales
                or node.exogenous != noise.exogenous
                or node.noise_response != noise.response
                or node.pipeline != pipeline
            ):
                raise ValueError("parent graph frozen runtime differs from retained realization")
            derived.append(JointPlanOwner(("graph", f"node:{i}"), "node", ("plans", i)))
            derived.extend(
                plan.owners(("plans", i))
                if isinstance(plan, ParentFunctionPlan)
                else plan_owners(plan, ("plans", i))
            )
        if tuple(derived) != self.owners:
            raise ValueError("parent graph typed ownership differs from plan tree")
        expected = tuple(owner.operation for owner in self.owners)
        if (
            tuple(
                span.operation
                for span in self.registry.operations
                if span.operation[:1] == ("graph",)
            )
            != expected
        ):
            raise ValueError("parent graph operation ownership differs")
        for plan in self.plans:
            plan.validate(self.registry.decisions)
        if type(self.raw) is not tuple or any(
            type(r) is not RawDrawRecord or r.sequence != i for i, r in enumerate(self.raw)
        ):
            raise ValueError("parent graph raw sequence differs")
        previous = 0
        for receipt in self.calibrations:
            if (
                type(receipt) is not ParentCalibrationReceipt
                or receipt.version != "parent-calibration-receipt-v1"
                or not 0 <= receipt.node < len(self.plans)
                or receipt.parents != self.request.parents[receipt.node]
                or not previous <= receipt.raw_start < receipt.raw_stop <= len(self.raw)
            ):
                raise ValueError("parent calibration receipt binding differs")
            owner = ("graph", "construction", f"node:{receipt.node}", receipt.role)
            if any(
                r.stream != "calibration" or r.owner[:4] != owner
                for r in self.raw[receipt.raw_start : receipt.raw_stop]
            ):
                raise ValueError("parent calibration raw provenance differs")
            previous = receipt.raw_stop
        if (
            self.runtime.task != self.task
            or self.runtime.observed != self.request.observed
            or self.runtime.target != self.request.target
        ):
            raise ValueError("parent graph runtime task or readouts differ")
        if tuple(n.parents for n in self.runtime.nodes) != self.request.parents:
            raise ValueError("parent graph ancestry differs")
