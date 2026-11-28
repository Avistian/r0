"""Admit and reconstruct recorded oblivious-tree fits without refitting query data."""

from dataclasses import fields as dataclass_fields
from typing import Any, cast

from r0.modules.priors.lib.causal_limits import CausalLimits
from r0.modules.priors.lib.frozen_function_decoding import frozen_function_from_dict
from r0.modules.priors.lib.frozen_function_primitives import FrozenTreeFunction
from r0.modules.priors.lib.joint_function_admission import admit_tree_function
from r0.modules.priors.lib.joint_function_calibration import FunctionCalibrationSample
from r0.modules.priors.lib.joint_function_decoding import joint_function_plan_from_dict
from r0.modules.priors.lib.joint_sampling_records import JointSamplingRecord
from r0.modules.priors.lib.joint_tree_realization import JointTreeFunctionRealization


def _record(value: Any, record_type: type[Any]) -> dict[str, Any]:
    if type(value) is not dict:
        raise ValueError("tree records must be JSON objects")
    result = cast(dict[str, Any], value)
    if set(result) - {field.name for field in dataclass_fields(record_type)}:
        raise ValueError("tree record contains unknown fields")
    return dict(result)


def _array(value: Any, count: int) -> tuple[Any, ...]:
    if type(value) not in (list, tuple) or len(value) != count:
        raise ValueError("tree arrays must match their exact planned geometry")
    return tuple(value)


def _table(value: Any, rows: int, columns: int) -> tuple[tuple[Any, ...], ...]:
    return tuple(_array(row, columns) for row in _array(value, rows))


def decode_joint_tree(
    data: dict[str, Any],
    decisions: JointSamplingRecord,
    limits: CausalLimits,
) -> JointTreeFunctionRealization:
    """The enclosing public decoder bounds the entire raw payload before this call."""
    fields = _record(data, JointTreeFunctionRealization)
    plan = joint_function_plan_from_dict(fields["plan"], decisions=decisions, limits=limits)
    admit_tree_function(plan, decisions, limits)
    parameters = dict(plan.parameters)
    trees, depth = int(parameters["trees"]), int(parameters["depth"])
    i, o = plan.input_width, plan.output_width
    calibration = _record(fields["calibration"], FunctionCalibrationSample)
    if calibration.get("version") != "joint-function-calibration-v1":
        raise ValueError("unsupported function calibration version")
    calibration["sample"] = _table(calibration["sample"], plan.calibration_rows, i)
    fields["calibration"] = FunctionCalibrationSample(**calibration)
    fields["plan"] = plan
    fields["feature_weights"] = _array(fields["feature_weights"], i)
    fields["features"] = _table(fields["features"], trees, depth)
    fields["threshold_rows"] = _table(fields["threshold_rows"], trees, depth)
    fields["leaf_gaussian"] = tuple(
        _table(table, 2**depth, o) for table in _array(fields["leaf_gaussian"], trees)
    )
    equation = _record(fields["function"], FrozenTreeFunction)
    if equation.get("version") != "frozen-oblivious-tree-function-v1":
        raise ValueError("tree construction requires a frozen oblivious-tree equation")
    for name in ("features", "thresholds"):
        _table(equation[name], trees, depth)
    for table in _array(equation["leaves"], trees):
        _table(table, 2**depth, o)
    fields["function"] = frozen_function_from_dict(equation, limits=limits)
    result = JointTreeFunctionRealization(**fields)
    result.validate(decisions)
    return result
