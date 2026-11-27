"""Admit source function constructions and retained metadata before raw arrays."""

from dataclasses import fields, is_dataclass
from typing import Any, cast

from r0.modules.priors.lib.causal_limits import CausalLimits
from r0.modules.priors.lib.joint_function_plans import JointFunctionPlan
from r0.modules.priors.lib.joint_matrix_realization import matrix_realization_parameters
from r0.modules.priors.lib.joint_sampling_records import JointSamplingRecord


def function_record_cost(value: Any, limits: CausalLimits) -> tuple[int, int]:
    """Count serialized numeric fields and estimated storage without copying records."""
    numbers, storage, nodes = 0, 0, 0

    def visit(item: Any, depth: int) -> None:
        nonlocal numbers, storage, nodes
        kind = cast(type[Any], type(item))
        record_fields = fields(item) if is_dataclass(item) and not isinstance(item, type) else ()
        nodes += 1
        if depth > 32 or nodes > 16 * limits.parameters + 64 * limits.nodes:
            raise ValueError("function metadata traversal limit exceeded")
        if kind in (int, float, bool):
            numbers += 1
            storage += 64
        elif kind is str:
            text = cast(str, item)
            if len(text) > limits.working_bytes:
                raise ValueError("function metadata text limit exceeded")
            storage += len(text.encode("utf-8"))
        elif kind is tuple or record_fields:
            storage += 32
        elif item is not None:
            raise ValueError("function metadata requires frozen records and primitive values")
        if numbers > limits.parameters or storage > limits.working_bytes:
            raise ValueError("function metadata parameter or storage limit exceeded")
        if kind is tuple:
            for child in cast(tuple[Any, ...], item):
                visit(child, depth + 1)
        elif record_fields:
            for field in record_fields:
                visit(field.name, depth + 1)
                visit(getattr(item, field.name), depth + 1)

    visit(value, 0)
    return numbers, storage


def admit_matrix_function(
    plan: JointFunctionPlan, decisions: JointSamplingRecord, limits: CausalLimits
) -> dict[str, int]:
    """Conservatively include the complete ledger, both stored plans and raw/derived copies."""
    if plan.family not in ("lin", "quad"):
        raise ValueError("matrix function realization branch is not implemented")
    matrix = plan.matrices[0]
    metadata, metadata_bytes = function_record_cost((plan, decisions, matrix), limits)
    frozen = plan.input_width * plan.output_width
    permutation = 0
    if plan.family == "quad":
        selected = min(plan.input_width, 20)
        frozen = 1 + selected + plan.output_width * (selected + 1) ** 2
        permutation = plan.input_width if plan.input_width > 20 else 0
    parameters = metadata + matrix_realization_parameters(matrix) + frozen + permutation
    report = limits.admit_dimensions(
        (max(plan.input_width, plan.output_width),),
        ((),),
        rows=matrix.batch * matrix.rows,
        calibration_rows=0,
        features=plan.input_width,
        outputs=plan.output_width,
        extra_parameters=parameters,
        extra_working_channels=8 * (matrix.rows + matrix.columns),
    )
    # Numeric storage is already covered by the conservative 128 bytes/parameter;
    # add record containers, keys and strings without double-counting those leaves.
    report["estimated_working_bytes"] += metadata_bytes - 64 * metadata
    if report["estimated_working_bytes"] > limits.working_bytes:
        raise ValueError("estimated function working memory exceeds declared limit")
    return report


def admit_rff_function(
    plan: JointFunctionPlan, decisions: JointSamplingRecord, limits: CausalLimits
) -> dict[str, int]:
    if plan.family != "gp":
        raise ValueError("Fourier realization branch is not implemented")
    i, o, k = plan.input_width, plan.output_width, plan.prior.frequencies
    metadata, metadata_bytes = function_record_cost((plan, decisions), limits)
    product = dict(plan.parameters)["product_kernel"] is True
    raw = k * ((i if product else 1) + 1 + o)
    if not product:
        assert plan.weights is not None
        raw += i * k + i * i + plan.weights.realization_parameter_count
    frozen = k * (i + 1 + o)
    report = limits.admit_dimensions(
        (max(i, o),),
        ((),),
        rows=k,
        calibration_rows=0,
        features=i,
        outputs=o,
        extra_parameters=metadata + raw + frozen,
        extra_working_channels=8 * i + 4 * o + 4,
    )
    report["estimated_working_bytes"] += metadata_bytes - 64 * metadata
    if report["estimated_working_bytes"] > limits.working_bytes:
        raise ValueError("estimated Fourier working memory exceeds declared limit")
    return report


def admit_tree_function(
    plan: JointFunctionPlan, decisions: JointSamplingRecord, limits: CausalLimits
) -> dict[str, int]:
    if plan.family != "tree":
        raise ValueError("tree realization requires a tree plan")
    i, o, rows = plan.input_width, plan.output_width, plan.calibration_rows
    values = dict(plan.parameters)
    trees, depth = int(values["trees"]), int(values["depth"])
    leaves = trees * 2**depth * o
    metadata, metadata_bytes = function_record_cost((plan, decisions), limits)
    # Sample, masses, two retained index tables, raw leaves and the frozen tree.
    raw = rows * i + i + 2 * trees * depth + leaves
    frozen = 1 + 2 * trees * depth + leaves
    report = limits.admit_dimensions(
        (max(i, o),),
        ((),),
        rows=rows,
        calibration_rows=0,
        features=i,
        outputs=o,
        extra_parameters=metadata + raw + frozen,
        extra_working_channels=8 * i + 4 * trees * depth + 4 * trees * o,
    )
    report["estimated_working_bytes"] += metadata_bytes - 64 * metadata + 256
    if report["estimated_working_bytes"] > limits.working_bytes:
        raise ValueError("estimated tree fitting memory exceeds declared limit")
    return report


def admit_discretization_function(
    plan: JointFunctionPlan, decisions: JointSamplingRecord, limits: CausalLimits
) -> dict[str, int]:
    if plan.family != "disc":
        raise ValueError("discretization realization requires a discretization plan")
    i, o, rows = plan.input_width, plan.output_width, plan.calibration_rows
    count = plan.center_count
    assert count is not None
    matrix = plan.matrices[0]
    metadata, metadata_bytes = function_record_cost((plan, decisions, matrix), limits)
    # Retain the sample, full permutation, matrix construction, targets and equation.
    raw = rows * i + (rows if rows > 1 else 0) + matrix_realization_parameters(matrix)
    frozen = count * (i + o) + i * o + 1
    report = limits.admit_dimensions(
        (max(i, o),),
        ((),),
        rows=max(rows, o),
        calibration_rows=0,
        features=i,
        outputs=o,
        extra_parameters=metadata + raw + frozen,
        extra_working_channels=8 * (i + o) + 4 * count,
    )
    report["estimated_working_bytes"] += metadata_bytes - 64 * metadata + 256
    if report["estimated_working_bytes"] > limits.working_bytes:
        raise ValueError("estimated discretization fitting memory exceeds declared limit")
    return report
