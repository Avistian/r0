"""Aggregate typed converter work and storage before allocating construction arrays."""

from dataclasses import replace

from r0.modules.priors.lib.converted_graph_planning import ConvertedGraphPlan
from r0.modules.priors.lib.converted_graph_request import ConvertedGraphRequest
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
from r0.modules.priors.lib.joint_graph_admission import admit_joint_graph
from r0.modules.priors.lib.joint_operation_registry import JointOperationRegistry
from r0.modules.priors.lib.joint_root_admission import admit_root_realization
from r0.modules.priors.lib.raw_draw_records import RawDrawLimits


def admit_converted_construction(
    request: ConvertedGraphRequest, plan: ConvertedGraphPlan, registry: JointOperationRegistry
) -> RawDrawLimits:
    geometry = request.geometry()
    metadata, storage = function_record_cost((request, plan, registry), request.limits)
    child = geometry.graph_request(
        plan.topology.parents, plan.layout, plan.widths, limits=request.limits
    )
    child = replace(child, authored_pipelines=())
    core = admit_joint_graph(child, plan.mechanisms, registry)
    parameters, working = (
        core.planned_parameters + 4 * metadata,
        core.planned_working_bytes + 4 * storage,
    )
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
    rows = max(request.rows, request.calibration_rows, 256)
    # Independent pipeline samples, constants, live typed prefixes and noise factors.
    numbers = 8 * (request.calibration_rows + 64) * sum(plan.widths.widths) + 8 * sum(
        w * w for w in plan.widths.widths
    )
    parameters += numbers
    working += 128 * numbers + 256 * rows * sum(plan.widths.widths)
    for converter in plan.categories:
        if converter is None:
            continue
        k, d, n = len(converter.vocabulary), converter.width, converter.calibration_rows
        numbers = 8 * (n * d + k * d + k + n)
        parameters += numbers
        working += 128 * numbers + 128 * rows * k * (d + 8)
        reports: list[dict[str, int]] = []
        if converter.embeddings is not None:
            reports.append(
                admit_root_realization(converter.embeddings, registry.decisions, request.limits)
            )
        if converter.function is not None:
            reports.append(
                admitters[converter.function.family](
                    converter.function, registry.decisions, request.limits
                )
            )
        for report in reports:
            parameters += 4 * report["parameters"]
            working += 4 * report["estimated_working_bytes"]
    remaining = request.limits.parameters - parameters
    remaining_bytes = request.limits.working_bytes - working
    raw = geometry.mechanism_raw_limits()
    raw_bytes = min(raw.working_bytes, remaining_bytes // 2, 32 * remaining)
    if remaining <= 0 or raw_bytes <= 2 * raw.state_bytes:
        raise ValueError("aggregate typed converter construction exceeds limits before allocation")
    return RawDrawLimits(
        raw.draws, min(raw.numbers, remaining // 2), raw_bytes, min(raw.state_bytes, 1024**2)
    )
