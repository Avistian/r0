"""Aggregate admission for every retained parent mechanism and ancestral fit."""

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
from r0.modules.priors.lib.joint_graph_admission import JointGraphAdmission
from r0.modules.priors.lib.joint_graph_request import JointGraphRequest
from r0.modules.priors.lib.joint_operation_registry import JointOperationRegistry
from r0.modules.priors.lib.joint_root_admission import (
    admit_root_realization,
    function_evaluation_channels,
)
from r0.modules.priors.lib.joint_root_plans import JointRootPlan
from r0.modules.priors.lib.parent_composition_plans import ParentFunctionPlan
from r0.modules.priors.lib.raw_draw_records import RawDrawLimits


def admit_parent_graph(
    request: JointGraphRequest,
    plans: tuple[JointRootPlan | ParentFunctionPlan, ...],
    registry: JointOperationRegistry,
) -> JointGraphAdmission:
    limits = request.limits
    count, storage = function_record_cost((request, plans, registry), limits)
    parameters, working = 4 * count, 4 * storage
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
    work_channels = 0
    member_channels = 0
    for plan in plans:
        if isinstance(plan, JointRootPlan):
            reports = (admit_root_realization(plan, registry.decisions, limits),)
            functions = (plan.function,)
        else:
            functions = plan.functions
            reports = tuple(admitters[p.family](p, registry.decisions, limits) for p in functions)
            member_channels = max(member_channels, len(functions) * plan.output_width)
        for report in reports:
            parameters += 4 * report["parameters"]
            working += 4 * report["estimated_working_bytes"]
        work_channels = max(work_channels, sum(function_evaluation_channels(p) for p in functions))
    total = sum(request.widths)
    fan_in = max(sum(request.widths[p] for p in incoming) for incoming in request.parents)
    pipeline = max((p.working_extra_channels for p in request.authored_pipelines if p), default=0)
    rows = max(request.task.rows, request.calibration_rows, 256)
    working += (
        16 * rows * (48 * total + 4 * fan_in + 8 * member_channels + work_channels + pipeline + 8)
    )
    working += 512 * total * total + 512 * total + 32 * min(request.raw_limits.state_bytes, 1024**2)
    parameters += 128 * len(plans) + 8 * member_channels
    remaining = limits.parameters - parameters
    available = limits.working_bytes - working
    raw_bytes = min(request.raw_limits.working_bytes, available // 2, 32 * remaining)
    if remaining <= 0 or raw_bytes <= 2 * request.raw_limits.state_bytes:
        raise ValueError("aggregate parent graph construction limit exceeded")
    raw = RawDrawLimits(
        request.raw_limits.draws,
        min(request.raw_limits.numbers, remaining // 2),
        raw_bytes,
        min(request.raw_limits.state_bytes, 1024**2),
    )
    return JointGraphAdmission(parameters, working, raw)
