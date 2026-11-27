"""Aggregate root construction, retained calibration and function evaluation bounds."""

from r0.modules.priors.lib.causal_limits import CausalLimits
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
from r0.modules.priors.lib.joint_root_plans import JointRootPlan
from r0.modules.priors.lib.joint_sampling_records import JointSamplingRecord


def function_evaluation_channels(plan: JointFunctionPlan) -> int:
    """Conservative simultaneous intermediate-array channels for emitted/fit rows."""
    i, o = plan.input_width, plan.output_width
    values = dict(plan.parameters)
    if plan.family == "lin":
        return 8 * (i + o)
    if plan.family == "quad":
        return 8 * o * (min(i, 20) + 1) + 8 * (i + o)
    if plan.family == "tree":
        return 8 * int(values["trees"]) * (int(values["depth"]) + o) + 8 * (i + o)
    if plan.family == "gp":
        return 8 * plan.prior.frequencies + 8 * (i + o)
    if plan.family in ("disc", "em"):
        count = plan.center_count if plan.family == "disc" else int(values["components"])
        assert count is not None
        return 16 * count * i + 12 * count + 16 * (i + o)
    if plan.family == "prod":
        return sum(function_evaluation_channels(child) for child in plan.children) + 8 * o
    return 24 * (i + sum(matrix.rows for matrix in plan.matrices))


def admit_root_realization(
    plan: JointRootPlan, decisions: JointSamplingRecord, limits: CausalLimits
) -> dict[str, int]:
    admitters = {
        "lin": admit_matrix_function,
        "quad": admit_matrix_function,
        "tree": admit_tree_function,
        "disc": admit_discretization_function,
        "gp": admit_rff_function,
        "em": admit_em_function,
        "mlp": admit_mlp_function,
        "prod": admit_product_function,
    }
    function = admitters[plan.function.family](plan.function, decisions, limits)
    metadata, storage = function_record_cost((plan, decisions), limits)
    rows, width = plan.calibration_rows, plan.width
    parameters = 16 + rows * (3 * width + int(plan.family == "ball"))
    if plan.family == "covariance":
        assert plan.weights is not None
        parameters += 2 * width**2 + plan.weights.realization_parameter_count
    report = limits.admit_dimensions(
        (width,),
        ((),),
        rows=rows,
        calibration_rows=0,
        features=width,
        outputs=width,
        extra_parameters=metadata + parameters,
        extra_working_channels=24 * width + function_evaluation_channels(plan.function),
    )
    report["parameters"] += function["parameters"]
    report["estimated_working_bytes"] += (
        function["estimated_working_bytes"] + storage - 64 * metadata + 1024
    )
    if report["parameters"] > limits.parameters:
        raise ValueError("aggregate root parameter limit exceeded")
    if report["estimated_working_bytes"] > limits.working_bytes:
        raise ValueError("aggregate root working memory limit exceeded")
    return report
