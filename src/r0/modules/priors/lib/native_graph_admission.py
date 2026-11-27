"""Aggregate all native construction fits, retained variates and prefix buffers."""

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
from r0.modules.priors.lib.joint_function_plans import JointFunctionPlan
from r0.modules.priors.lib.joint_operation_registry import JointOperationRegistry
from r0.modules.priors.lib.joint_root_admission import (
    admit_root_realization,
    function_evaluation_channels,
)
from r0.modules.priors.lib.joint_root_plans import JointRootPlan
from r0.modules.priors.lib.native_domain_prior import NativeDomainPlan
from r0.modules.priors.lib.native_graph_request import NativeGraphRequest
from r0.modules.priors.lib.parent_composition_plans import ParentFunctionPlan
from r0.modules.priors.lib.raw_draw_records import RawDrawLimits


def admit_native_construction(
    request: NativeGraphRequest,
    domains: tuple[NativeDomainPlan, ...],
    mechanisms: tuple[JointRootPlan | JointFunctionPlan | ParentFunctionPlan, ...],
    registry: JointOperationRegistry,
) -> RawDrawLimits:
    limits = request.limits
    parameters, storage = function_record_cost((request, domains, mechanisms, registry), limits)
    parameters, working = 4 * parameters, 4 * storage
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
    for mechanism in mechanisms:
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
    total = sum(d.score_width + d.state_width + 2 for d in domains)
    fan_in = max(sum(domains[p].state_width for p in parents) for parents in request.parents)
    rows = max(request.rows, request.calibration_rows, 256)
    working += 16 * rows * (48 * total + 4 * fan_in + work_channels + 8)
    # Typed states, emission probabilities/uniforms and serialized scalar copies.
    working += 256 * rows * total
    working += 512 * total * total + 512 * total
    working += 32 * min(request.raw_limits.state_bytes, 1024**2)
    parameters += 128 * len(domains)
    remaining_parameters = limits.parameters - parameters
    remaining_bytes = limits.working_bytes - working
    raw_bytes = min(
        request.raw_limits.working_bytes, remaining_bytes // 2, 32 * remaining_parameters
    )
    if remaining_parameters <= 0 or raw_bytes <= 2 * request.raw_limits.state_bytes:
        raise ValueError("aggregate native construction exceeds parameter or working memory limit")
    return RawDrawLimits(
        request.raw_limits.draws,
        min(request.raw_limits.numbers, remaining_parameters // 2),
        raw_bytes,
        min(request.raw_limits.state_bytes, 1024**2),
    )
