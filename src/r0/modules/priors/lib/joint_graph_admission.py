"""Conservative whole-construction bounds before sampling any raw arrays."""

from dataclasses import dataclass

from r0.modules.priors.lib.joint_function_admission import (
    admit_discretization_function,
    admit_em_function,
    admit_matrix_function,
    admit_mlp_function,
    admit_product_function,
    admit_rff_function,
    admit_tree_function,
    function_record_cost,
)
from r0.modules.priors.lib.joint_graph_plans import JointGraphPlan
from r0.modules.priors.lib.joint_graph_request import JointGraphRequest
from r0.modules.priors.lib.joint_operation_registry import JointOperationRegistry
from r0.modules.priors.lib.joint_root_admission import (
    admit_root_realization,
    function_evaluation_channels,
)
from r0.modules.priors.lib.joint_root_plans import JointRootPlan
from r0.modules.priors.lib.parent_composition_plans import ParentFunctionPlan
from r0.modules.priors.lib.raw_draw_records import RawDrawLimits


@dataclass(frozen=True, slots=True)
class JointGraphAdmission:
    planned_parameters: int
    planned_working_bytes: int
    raw_limits: RawDrawLimits
    version: str = "joint-graph-admission-v1"


def admit_joint_graph(
    request: JointGraphRequest, plan: JointGraphPlan, registry: JointOperationRegistry
) -> JointGraphAdmission:
    """Sum complete child bounds; deliberately retain their duplicated ledger costs.

    Four copies cover realizations, runtime, validation and replay. Raw recording
    gets only the remaining budget. These are estimates, not measured RSS.
    """
    limits = request.limits
    metadata, storage = function_record_cost((request, plan, registry, plan.owners()), limits)
    admitters = {
        "lin": admit_matrix_function,
        "quad": admit_matrix_function,
        "gp": admit_rff_function,
        "tree": admit_tree_function,
        "disc": admit_discretization_function,
        "em": admit_em_function,
        "mlp": admit_mlp_function,
        "prod": admit_product_function,
    }
    parameters, working = 4 * metadata, 4 * storage
    work_channels = 0
    for node in plan.nodes:
        mechanism = node.mechanism
        if isinstance(mechanism, JointRootPlan):
            report = admit_root_realization(mechanism, registry.decisions, limits)
            function = mechanism.function
        elif isinstance(mechanism, ParentFunctionPlan):
            reports = tuple(
                admitters[c.family](c, registry.decisions, limits) for c in mechanism.functions
            )
            parameters += (
                4 * sum(r["parameters"] for r in reports)
                + 8 * len(mechanism.functions) * mechanism.output_width
            )
            working += 4 * sum(r["estimated_working_bytes"] for r in reports)
            # One fit receipt can retain every parent reference, for every member.
            # Charge serialized/replay copies even though Python shares the tuple.
            receipt_numbers = len(mechanism.functions) * (len(mechanism.parent_widths) + 16)
            parameters += 4 * receipt_numbers
            working += 512 * receipt_numbers
            work_channels = max(
                work_channels,
                sum(function_evaluation_channels(c) for c in mechanism.functions)
                + 8 * len(mechanism.functions) * mechanism.output_width,
            )
            continue
        else:
            report = admitters[mechanism.family](mechanism, registry.decisions, limits)
            function = mechanism
        parameters += 4 * report["parameters"]
        working += 4 * report["estimated_working_bytes"]
        work_channels = max(work_channels, function_evaluation_channels(function))
    total = sum(request.widths)
    fan_in = max(sum(request.widths[p] for p in incoming) for incoming in request.parents)
    pipeline_work = max(
        (p.working_extra_channels for p in request.authored_pipelines if p), default=0
    )
    rows = max(request.task.rows, request.calibration_rows, 256)
    # Prefix states/readouts, parent concatenation, covariance/root draws and
    # certificates remain live while the receiver is fitted. Root/function costs
    # above already include each retained calibration matrix and derived equation.
    working += 16 * rows * (48 * total + 4 * fan_in + work_channels + pipeline_work + 8)
    working += 512 * total * total + 512 * total
    # Four streams, original/final snapshots, transaction copies and JSON parsing.
    working += 32 * min(request.raw_limits.state_bytes, 1024**2)
    parameters += 128 * len(plan.nodes)
    remaining_parameters = limits.parameters - parameters
    remaining_bytes = limits.working_bytes - working
    if remaining_parameters <= 0 or remaining_bytes <= 2 * request.raw_limits.state_bytes:
        raise ValueError("aggregate graph construction parameter or working memory limit exceeded")
    # Each ledger numeric leaf costs at least 96 bytes; its per-record metadata
    # charge is larger. Reserve a second copy for full graph replay/validation.
    raw_bytes = min(
        request.raw_limits.working_bytes, remaining_bytes // 2, 32 * remaining_parameters
    )
    if raw_bytes <= 2 * request.raw_limits.state_bytes:
        raise ValueError("aggregate graph raw draw working memory limit exceeded")
    raw = RawDrawLimits(
        request.raw_limits.draws,
        min(request.raw_limits.numbers, remaining_parameters // 2),
        raw_bytes,
        min(request.raw_limits.state_bytes, 1024**2),
    )
    return JointGraphAdmission(parameters, working, raw)
