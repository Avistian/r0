"""Retained sampled graph constructions and explicit, executable provenance replay."""

from dataclasses import dataclass

from r0.modules.priors.lib.functional_causal import FunctionalCausalRecipe
from r0.modules.priors.lib.functional_graph_admission import admit_functional_graph
from r0.modules.priors.lib.generator_snapshot import GeneratorSnapshot
from r0.modules.priors.lib.joint_graph_admission import JointGraphAdmission, admit_joint_graph
from r0.modules.priors.lib.joint_graph_plans import JointGraphPlan
from r0.modules.priors.lib.joint_graph_request import JointGraphNoise, JointGraphRequest
from r0.modules.priors.lib.joint_operation_registry import JointOperationRegistry
from r0.modules.priors.lib.joint_plan_ownership import JointPlanOwner
from r0.modules.priors.lib.joint_root_realization import (
    JointRootRealization,
    RootFunctionRealization,
)
from r0.modules.priors.lib.raw_draw_records import RawDrawRecord


@dataclass(frozen=True, slots=True)
class GraphCalibrationReceipt:
    node: int
    identity: str
    parents: tuple[int, ...]
    raw_start: int
    raw_stop: int
    input_digest: str
    version: str = "graph-calibration-receipt-v1"


@dataclass(frozen=True, slots=True)
class SampledFunctionalGraph:
    request: JointGraphRequest
    plan: JointGraphPlan
    registry: JointOperationRegistry
    owners: tuple[JointPlanOwner, ...]
    realizations: tuple[JointRootRealization | RootFunctionRealization, ...]
    calibrations: tuple[GraphCalibrationReceipt, ...]
    raw: tuple[RawDrawRecord, ...]
    initial: tuple[GeneratorSnapshot, ...]
    final: tuple[GeneratorSnapshot, ...]
    admission: JointGraphAdmission
    runtime: FunctionalCausalRecipe
    version: str = "sampled-functional-graph-v1"

    def validate(self) -> None:
        """Check the envelope, then regenerate ALL choices, raw draws and fits.

        Structural equality alone cannot establish that a fitted population came
        through its declared ancestors. Replay uses retained bounded JSON states,
        never a caller generator. It also checks event-free owners and raw roles.
        """
        self.replay()

    def replay(self) -> "SampledFunctionalGraph":
        from r0.modules.priors.lib.joint_graph_sampling import sample_joint_graph

        self.validate_structure()
        streams = tuple(
            s.restore(byte_limit=self.request.raw_limits.state_bytes) for s in self.initial
        )
        result = sample_joint_graph(
            self.request,
            named_rng=streams[0],
            raw_rng=streams[1],
            calibration_rng=streams[2],
            fit_rng=streams[3],
        )
        if result != self:
            raise ValueError("sampled graph replay differs from retained construction or runtime")
        return result

    def validate_structure(self) -> None:
        """Array-free envelope admission precedes reconstructing any fit or RNG state."""
        if (
            self.version != "sampled-functional-graph-v1"
            or type(self.request) is not JointGraphRequest
        ):
            raise ValueError("sampled graph requires its version and frozen request")
        if (
            type(self.runtime) is not FunctionalCausalRecipe
            or type(self.plan) is not JointGraphPlan
        ):
            raise ValueError("sampled graph requires its frozen plan and runtime")
        if (
            type(self.realizations) is not tuple
            or len(self.realizations) != len(self.plan.nodes)
            or type(self.initial) is not tuple
            or type(self.final) is not tuple
            or len(self.initial) != 4
            or len(self.final) != 4
            or any(type(s) is not GeneratorSnapshot for s in (*self.initial, *self.final))
        ):
            raise ValueError(
                "sampled graph requires aligned realizations and four stream snapshots"
            )
        # This counts the entire envelope, including raw data and all duplicate
        # serialized plans/runtime equations, before any ndarray construction.
        admit_functional_graph(
            self.runtime.nodes,
            self.request.limits,
            rows=max(self.request.task.rows, self.request.calibration_rows),
            metadata=(self,),
            features=self.request.task.features,
            outputs=len(self.request.target),
        )
        self.plan.validate(self.registry, owners=self.owners)
        expected = admit_joint_graph(self.request, self.plan, self.registry)
        if expected != self.admission:
            raise ValueError("sampled graph admission differs from request and plan")
        if (
            self.runtime.task != self.request.task
            or self.runtime.observed != self.request.observed
            or self.runtime.target != self.request.target
            or self.runtime.limits != self.request.limits
            or tuple(n.parents for n in self.plan.nodes) != self.request.parents
            or tuple(n.width for n in self.plan.nodes) != self.request.widths
            or self.plan.calibration_rows != self.request.calibration_rows
        ):
            raise ValueError("sampled graph runtime or plan differs from request")
        if type(self.raw) is not tuple or any(
            type(r) is not RawDrawRecord or r.sequence != i for i, r in enumerate(self.raw)
        ):
            raise ValueError("sampled graph raw records require contiguous ordered sequences")
        if type(self.calibrations) is not tuple or any(
            type(r) is not GraphCalibrationReceipt for r in self.calibrations
        ):
            raise ValueError("sampled graph requires frozen calibration receipts")
        if len(self.runtime.nodes) != len(self.plan.nodes):
            raise ValueError("sampled graph runtime node count differs from plan")
        expected_receivers: list[int] = []
        for i, (planned, realized, runtime) in enumerate(
            zip(self.plan.nodes, self.realizations, self.runtime.nodes, strict=True)
        ):
            if realized.plan != planned.mechanism:
                raise ValueError("sampled graph realization differs from its owned plan")
            realized.validate(self.registry.decisions)
            if isinstance(realized, JointRootRealization):
                function, root = realized.function.function, realized.calibration.base
            else:
                function, root = realized.function, None
                if getattr(realized, "calibration", None) is not None:
                    expected_receivers.append(i)
            noise = self.request.noise[i] if self.request.noise else JointGraphNoise()
            pipeline = (
                self.request.authored_pipelines[i] if self.request.authored_pipelines else None
            )
            if (
                runtime.parents != planned.parents
                or runtime.function != function
                or runtime.root != root
                or runtime.noise_scale != noise.scales
                or runtime.exogenous != noise.exogenous
                or runtime.noise_response != noise.response
                or runtime.pipeline != pipeline
            ):
                raise ValueError("sampled graph runtime differs from retained construction")
        if tuple(r.node for r in self.calibrations) != tuple(expected_receivers):
            raise ValueError("sampled graph calibration receipts differ from fitted receivers")
        previous_stop = 0
        for receipt in self.calibrations:
            sample = getattr(self.realizations[receipt.node], "calibration", None)
            if (
                sample is None
                or receipt.version != "graph-calibration-receipt-v1"
                or receipt.parents != self.plan.nodes[receipt.node].parents
                or receipt.identity != sample.identity
                or receipt.input_digest != sample.digest
                or type(receipt.raw_start) is not int
                or type(receipt.raw_stop) is not int
                or not previous_stop <= receipt.raw_start < receipt.raw_stop <= len(self.raw)
            ):
                raise ValueError("sampled graph calibration receipt binding differs")
            owner = ("graph", "construction", f"node:{receipt.node}", "function-fit")
            if any(
                r.stream != "calibration" or r.owner[:4] != owner
                for r in self.raw[receipt.raw_start : receipt.raw_stop]
            ):
                raise ValueError("sampled graph calibration receipt raw provenance differs")
            previous_stop = receipt.raw_stop
