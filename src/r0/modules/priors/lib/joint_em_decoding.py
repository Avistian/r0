"""Admit and bind saved EM anchors, raw Gaussian draws and fit assignments."""

from dataclasses import MISSING
from dataclasses import fields as dataclass_fields
from typing import Any, cast

from r0.modules.priors.lib.causal_limits import CausalLimits
from r0.modules.priors.lib.frozen_function_composites import FrozenEMFunction
from r0.modules.priors.lib.frozen_function_decoding import frozen_function_from_dict
from r0.modules.priors.lib.frozen_function_primitives import FrozenLinearFunction
from r0.modules.priors.lib.joint_em_realization import JointEMFunctionRealization
from r0.modules.priors.lib.joint_function_admission import admit_em_function
from r0.modules.priors.lib.joint_function_calibration import FunctionCalibrationSample
from r0.modules.priors.lib.joint_function_decoding import joint_function_plan_from_dict
from r0.modules.priors.lib.joint_matrix_decoding import joint_matrix_realization_from_dict
from r0.modules.priors.lib.joint_sampling_records import JointSamplingRecord
from r0.modules.priors.lib.weight_geometry import Matrix


def _record(value: Any, record_type: type[Any]) -> dict[str, Any]:
    if type(value) is not dict:
        raise ValueError("EM records must be JSON objects")
    result = cast(dict[str, Any], value)
    fields = dataclass_fields(record_type)
    required = {
        field.name
        for field in fields
        if field.default is MISSING and field.default_factory is MISSING
    }
    if set(result) - {field.name for field in fields} or required - set(result):
        raise ValueError("EM record contains unknown fields or lacks required fields")
    return dict(result)


def _array(value: Any, count: int) -> tuple[Any, ...]:
    if type(value) not in (list, tuple) or len(value) != count:
        raise ValueError("EM arrays must match their exact planned geometry")
    return tuple(value)


def _matrix(value: Any, rows: int, columns: int) -> Matrix:
    return tuple(_array(row, columns) for row in _array(value, rows))


def decode_joint_em(
    data: dict[str, Any],
    decisions: JointSamplingRecord,
    limits: CausalLimits,
) -> JointEMFunctionRealization:
    """The enclosing decoder admits the entire raw payload before this call."""
    fields = _record(data, JointEMFunctionRealization)
    plan = joint_function_plan_from_dict(fields["plan"], decisions=decisions, limits=limits)
    admit_em_function(plan, decisions, limits)
    i, o, rows = plan.input_width, plan.output_width, plan.calibration_rows
    count = int(dict(plan.parameters)["components"])
    calibration = _record(fields["calibration"], FunctionCalibrationSample)
    if calibration.get("version") != "joint-function-calibration-v1":
        raise ValueError("unsupported function calibration version")
    calibration["sample"] = _matrix(calibration["sample"], rows, i)
    fields["calibration"] = FunctionCalibrationSample(**calibration)
    fields["plan"] = plan
    fields["anchor_rows"] = _array(fields["anchor_rows"], count)
    fields["center_gaussian"] = _matrix(fields["center_gaussian"], count, i)
    fields["spread_gaussian"] = _array(fields["spread_gaussian"], count)
    fields["assignments"] = _matrix(fields["assignments"], rows, count)
    fields["matrix"] = joint_matrix_realization_from_dict(
        fields["matrix"], decisions=decisions, limits=limits
    )
    equation = _record(fields["function"], FrozenEMFunction)
    if equation.get("version") != "frozen-em-function-v1":
        raise ValueError("EM construction requires its frozen equation version")
    _matrix(equation["centers"], count, i)
    _array(equation["scales"], count)
    linear = _record(equation["linear"], FrozenLinearFunction)
    if linear.get("version") != "frozen-linear-function-v1":
        raise ValueError("EM construction requires a frozen linear map")
    _matrix(linear["matrix"], o, count)
    fields["function"] = frozen_function_from_dict(equation, limits=limits)
    result = JointEMFunctionRealization(**fields)
    result.validate(decisions)
    return result
