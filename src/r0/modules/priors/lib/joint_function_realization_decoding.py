"""Bound raw function constructions before reconstructing their immutable records."""

from dataclasses import fields as dataclass_fields
from typing import Any, cast

from r0.modules.priors.lib.causal_limits import CausalLimits
from r0.modules.priors.lib.frozen_function_decoding import frozen_function_from_dict
from r0.modules.priors.lib.frozen_function_primitives import (
    FrozenLinearFunction,
    FrozenQuadraticFunction,
)
from r0.modules.priors.lib.function_metadata import admit_function_metadata
from r0.modules.priors.lib.joint_function_admission import admit_matrix_function
from r0.modules.priors.lib.joint_function_decoding import joint_function_plan_from_dict
from r0.modules.priors.lib.joint_function_realization import JointMatrixFunctionRealization
from r0.modules.priors.lib.joint_matrix_decoding import joint_matrix_realization_from_dict
from r0.modules.priors.lib.joint_sampling_records import JointSamplingRecord

_DEFAULT_LIMITS = CausalLimits()


def joint_function_realization_from_dict(
    data: dict[str, Any],
    *,
    decisions: JointSamplingRecord,
    limits: CausalLimits = _DEFAULT_LIMITS,
) -> JointMatrixFunctionRealization:
    if type(decisions) is not JointSamplingRecord or type(limits) is not CausalLimits:
        raise ValueError(
            "function reconstruction requires frozen decisions and caller-owned limits"
        )
    admit_function_metadata(data, limits)
    if type(data) is not dict:
        raise ValueError("function realization must be a JSON object")
    if set(data) - {field.name for field in dataclass_fields(JointMatrixFunctionRealization)}:
        raise ValueError("function realization contains unknown fields")
    if data.get("version") != "joint-matrix-function-realization-v1":
        raise ValueError("unsupported joint function realization version")
    plan = joint_function_plan_from_dict(data["plan"], decisions=decisions, limits=limits)
    admit_matrix_function(plan, decisions, limits)
    count = plan.input_width if plan.family == "quad" and plan.input_width > 20 else 0
    permutation = data.get("permutation", ())
    if type(permutation) not in (list, tuple) or len(permutation) != count:
        raise ValueError("function raw permutation must match its exact source geometry")
    equation = data["function"]
    version = (
        "frozen-linear-function-v1" if plan.family == "lin" else "frozen-quadratic-function-v1"
    )
    if type(equation) is not dict:
        raise ValueError("frozen equation must be a JSON object")
    equation = cast(dict[str, Any], equation)
    if equation.get("version") != version:
        raise ValueError("frozen equation version differs from its function plan")
    matrix = joint_matrix_realization_from_dict(data["matrix"], decisions=decisions, limits=limits)
    function = frozen_function_from_dict(equation, limits=limits)
    if not isinstance(function, (FrozenLinearFunction, FrozenQuadraticFunction)):
        raise ValueError("matrix function realization requires a linear or quadratic equation")
    result = JointMatrixFunctionRealization(plan, matrix, function, tuple(permutation))
    result.validate(decisions)
    return result
