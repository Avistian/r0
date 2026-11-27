"""Bounded reconstruction of retained discretization fits and their linear targets."""

from dataclasses import MISSING
from dataclasses import fields as dataclass_fields
from typing import Any, cast

from r0.modules.priors.lib.causal_limits import CausalLimits
from r0.modules.priors.lib.frozen_function_composites import FrozenDiscretizationFunction
from r0.modules.priors.lib.frozen_function_decoding import frozen_function_from_dict
from r0.modules.priors.lib.frozen_function_primitives import FrozenLinearFunction
from r0.modules.priors.lib.joint_discretization_realization import (
    JointDiscretizationFunctionRealization,
)
from r0.modules.priors.lib.joint_function_admission import admit_discretization_function
from r0.modules.priors.lib.joint_function_calibration import FunctionCalibrationSample
from r0.modules.priors.lib.joint_function_decoding import joint_function_plan_from_dict
from r0.modules.priors.lib.joint_matrix_decoding import joint_matrix_realization_from_dict
from r0.modules.priors.lib.joint_sampling_records import JointSamplingRecord
from r0.modules.priors.lib.weight_geometry import Matrix


def _record(value: Any, record_type: type[Any]) -> dict[str, Any]:
    if type(value) is not dict:
        raise ValueError("discretization records must be JSON objects")
    result = cast(dict[str, Any], value)
    fields = dataclass_fields(record_type)
    required = {
        field.name
        for field in fields
        if field.default is MISSING and field.default_factory is MISSING
    }
    if set(result) - {field.name for field in fields} or required - set(result):
        raise ValueError("discretization record contains unknown fields or lacks required fields")
    return dict(result)


def _array(value: Any, count: int) -> tuple[Any, ...]:
    if type(value) not in (list, tuple) or len(value) != count:
        raise ValueError("discretization arrays must match their exact planned geometry")
    return tuple(value)


def _matrix(value: Any, rows: int, columns: int) -> Matrix:
    return tuple(_array(row, columns) for row in _array(value, rows))


def decode_joint_discretization(
    data: dict[str, Any],
    decisions: JointSamplingRecord,
    limits: CausalLimits,
) -> JointDiscretizationFunctionRealization:
    """The enclosing decoder admits the entire raw payload before this call."""
    fields = _record(data, JointDiscretizationFunctionRealization)
    plan = joint_function_plan_from_dict(fields["plan"], decisions=decisions, limits=limits)
    admit_discretization_function(plan, decisions, limits)
    i, o, rows = plan.input_width, plan.output_width, plan.calibration_rows
    count = plan.center_count
    assert count is not None
    calibration = _record(fields["calibration"], FunctionCalibrationSample)
    if calibration.get("version") != "joint-function-calibration-v1":
        raise ValueError("unsupported function calibration version")
    calibration["sample"] = _matrix(calibration["sample"], rows, i)
    fields["calibration"] = FunctionCalibrationSample(**calibration)
    fields["plan"] = plan
    fields["permutation"] = _array(fields["permutation"], rows if rows > 1 else 0)
    fields["targets"] = _matrix(fields["targets"], count, o)
    fields["matrix"] = joint_matrix_realization_from_dict(
        fields["matrix"], decisions=decisions, limits=limits
    )
    equation = _record(fields["function"], FrozenDiscretizationFunction)
    if equation.get("version") != "frozen-discretization-function-v1":
        raise ValueError("discretization construction requires its frozen equation version")
    _matrix(equation["centers"], count, i)
    linear = _record(equation["linear"], FrozenLinearFunction)
    if linear.get("version") != "frozen-linear-function-v1":
        raise ValueError("discretization construction requires a frozen linear map")
    _matrix(linear["matrix"], o, i)
    fields["function"] = frozen_function_from_dict(equation, limits=limits)
    result = JointDiscretizationFunctionRealization(**fields)
    result.validate(decisions)
    return result
