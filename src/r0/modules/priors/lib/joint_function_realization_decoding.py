"""Bound raw function constructions before reconstructing their immutable records."""

from dataclasses import fields as dataclass_fields
from typing import Any, cast

from r0.modules.priors.lib.causal_limits import CausalLimits
from r0.modules.priors.lib.frozen_function_composites import FrozenProductFunction
from r0.modules.priors.lib.frozen_function_decoding import frozen_function_from_dict
from r0.modules.priors.lib.frozen_function_primitives import (
    FrozenLinearFunction,
    FrozenQuadraticFunction,
)
from r0.modules.priors.lib.function_metadata import admit_function_metadata
from r0.modules.priors.lib.joint_discretization_decoding import decode_joint_discretization
from r0.modules.priors.lib.joint_discretization_realization import (
    JointDiscretizationFunctionRealization,
)
from r0.modules.priors.lib.joint_em_decoding import decode_joint_em
from r0.modules.priors.lib.joint_em_realization import JointEMFunctionRealization
from r0.modules.priors.lib.joint_function_admission import (
    admit_matrix_function,
    admit_product_function,
)
from r0.modules.priors.lib.joint_function_calibration import FunctionCalibrationSample
from r0.modules.priors.lib.joint_function_decoding import joint_function_plan_from_dict
from r0.modules.priors.lib.joint_function_realization import (
    JointMatrixFunctionRealization,
    JointProductFunctionRealization,
)
from r0.modules.priors.lib.joint_matrix_decoding import joint_matrix_realization_from_dict
from r0.modules.priors.lib.joint_rff_decoding import decode_joint_rff
from r0.modules.priors.lib.joint_rff_realization import JointRFFFunctionRealization
from r0.modules.priors.lib.joint_sampling_records import JointSamplingRecord
from r0.modules.priors.lib.joint_tree_decoding import decode_joint_tree
from r0.modules.priors.lib.joint_tree_realization import JointTreeFunctionRealization

_DEFAULT_LIMITS = CausalLimits()


def joint_function_realization_from_dict(
    data: dict[str, Any],
    *,
    decisions: JointSamplingRecord,
    limits: CausalLimits = _DEFAULT_LIMITS,
) -> (
    JointMatrixFunctionRealization
    | JointRFFFunctionRealization
    | JointTreeFunctionRealization
    | JointDiscretizationFunctionRealization
    | JointEMFunctionRealization
    | JointProductFunctionRealization
):
    if type(decisions) is not JointSamplingRecord or type(limits) is not CausalLimits:
        raise ValueError(
            "function reconstruction requires frozen decisions and caller-owned limits"
        )
    admit_function_metadata(data, limits)
    if type(data) is not dict:
        raise ValueError("function realization must be a JSON object")
    if data.get("version") == "joint-product-function-realization-v1":
        return _decode_product(data, decisions, limits)
    if data.get("version") == "joint-em-function-realization-v1":
        return decode_joint_em(data, decisions, limits)
    if data.get("version") == "joint-discretization-function-realization-v1":
        return decode_joint_discretization(data, decisions, limits)
    if data.get("version") == "joint-rff-function-realization-v1":
        return decode_joint_rff(data, decisions, limits)
    if data.get("version") == "joint-tree-function-realization-v1":
        return decode_joint_tree(data, decisions, limits)
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


def _decode_product(
    data: dict[str, Any],
    decisions: JointSamplingRecord,
    limits: CausalLimits,
) -> JointProductFunctionRealization:
    allowed = {field.name for field in dataclass_fields(JointProductFunctionRealization)}
    required = {"plan", "left", "right", "function", "version"}
    if set(data) - allowed or required - set(data):
        raise ValueError("product record has unknown fields or lacks required fields")
    plan = joint_function_plan_from_dict(data["plan"], decisions=decisions, limits=limits)
    admit_product_function(plan, decisions, limits)
    # Child plans are closed cheap selections; checking their ownership before
    # descent also prevents recursive payloads from entering this branch again.
    for name, expected in zip(("left", "right"), plan.children, strict=True):
        child = data[name]
        if type(child) is not dict or "plan" not in child:
            raise ValueError("product children must be realization objects")
        child = cast(dict[str, Any], child)
        child_plan = joint_function_plan_from_dict(
            child["plan"], decisions=decisions, limits=limits
        )
        if child_plan != expected:
            raise ValueError("product children differ from their ordered owned plans")
    left = joint_function_realization_from_dict(data["left"], decisions=decisions, limits=limits)
    right = joint_function_realization_from_dict(data["right"], decisions=decisions, limits=limits)
    cheap = (
        JointMatrixFunctionRealization,
        JointRFFFunctionRealization,
        JointTreeFunctionRealization,
        JointDiscretizationFunctionRealization,
    )
    if not isinstance(left, cheap) or not isinstance(right, cheap):
        raise ValueError("product requires nonrecursive cheap child realizations")
    calibration = None
    raw_calibration = data.get("calibration")
    fitted = any(child.family in ("tree", "disc") for child in plan.children)
    if fitted:
        if type(raw_calibration) is not dict:
            raise ValueError("fitted product requires a calibration record")
        fields = dict(cast(dict[str, Any], raw_calibration))
        if set(fields) != {field.name for field in dataclass_fields(FunctionCalibrationSample)}:
            raise ValueError("product calibration must retain exactly its required fields")
        if fields["version"] != "joint-function-calibration-v1":
            raise ValueError("unsupported product calibration version")
        sample = fields["sample"]
        if type(sample) not in (list, tuple) or len(sample) != plan.calibration_rows:
            raise ValueError("product calibration rows differ from planned geometry")
        if any(type(row) not in (list, tuple) or len(row) != plan.input_width for row in sample):
            raise ValueError("product calibration channels differ from planned geometry")
        fields["sample"] = tuple(tuple(row) for row in sample)
        calibration = FunctionCalibrationSample(**fields)
    elif raw_calibration is not None:
        raise ValueError("product of unfitted functions cannot own calibration")
    equation = frozen_function_from_dict(data["function"], limits=limits)
    if type(equation) is not FrozenProductFunction:
        raise ValueError("product construction requires a frozen product equation")
    result = JointProductFunctionRealization(plan, left, right, equation, calibration)
    result.validate(decisions)
    return result
