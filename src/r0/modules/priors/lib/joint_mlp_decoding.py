"""Admit MLP realization records before reconstructing and checking fitted stages."""

from dataclasses import MISSING
from dataclasses import fields as dataclass_fields
from typing import Any, cast

from r0.modules.priors.lib.calibrated_activations import CalibratedActivation
from r0.modules.priors.lib.causal_limits import CausalLimits
from r0.modules.priors.lib.frozen_function_composites import FrozenMLPFunction
from r0.modules.priors.lib.frozen_function_decoding import frozen_function_from_dict
from r0.modules.priors.lib.frozen_function_primitives import FrozenLinearFunction
from r0.modules.priors.lib.joint_activation_plans import JointActivationPlan
from r0.modules.priors.lib.joint_function_admission import admit_mlp_function
from r0.modules.priors.lib.joint_function_calibration import FunctionCalibrationSample
from r0.modules.priors.lib.joint_function_decoding import joint_function_plan_from_dict
from r0.modules.priors.lib.joint_matrix_decoding import joint_matrix_realization_from_dict
from r0.modules.priors.lib.joint_matrix_plans import JointMatrixPlan
from r0.modules.priors.lib.joint_mlp_realization import JointMLPFunctionRealization
from r0.modules.priors.lib.joint_sampling_records import JointSamplingRecord
from r0.modules.priors.lib.weight_geometry import Matrix


def _record(value: Any, record_type: type[Any]) -> dict[str, Any]:
    if type(value) is not dict:
        raise ValueError("MLP records must be JSON objects")
    result = cast(dict[str, Any], value)
    fields = dataclass_fields(record_type)
    required = {
        field.name
        for field in fields
        if field.default is MISSING and field.default_factory is MISSING
    }
    if set(result) - {field.name for field in fields} or required - set(result):
        raise ValueError("MLP record contains unknown fields or lacks required fields")
    return dict(result)


def _array(value: Any, count: int) -> tuple[Any, ...]:
    if type(value) not in (list, tuple) or len(value) != count:
        raise ValueError("MLP arrays must match their exact planned geometry")
    return tuple(value)


def _matrix(value: Any, rows: int, columns: int) -> Matrix:
    return tuple(_array(row, columns) for row in _array(value, rows))


def _sample(value: Any, rows: int, columns: int) -> FunctionCalibrationSample:
    fields = _record(value, FunctionCalibrationSample)
    if fields.get("version") != "joint-function-calibration-v1":
        raise ValueError("unsupported function calibration version")
    fields["sample"] = _matrix(fields["sample"], rows, columns)
    return FunctionCalibrationSample(**fields)


def decode_joint_mlp(
    data: dict[str, Any],
    decisions: JointSamplingRecord,
    limits: CausalLimits,
) -> JointMLPFunctionRealization:
    """The enclosing decoder admits the entire raw payload before this call."""
    fields = _record(data, JointMLPFunctionRealization)
    plan = joint_function_plan_from_dict(fields["plan"], decisions=decisions, limits=limits)
    admit_mlp_function(plan, decisions, limits)
    i, o, rows = plan.input_width, plan.output_width, plan.calibration_rows
    widths: list[int] = []
    width = i
    for child in plan.ordered_children:
        if isinstance(child, JointActivationPlan):
            widths.append(width)
        else:
            assert isinstance(child, JointMatrixPlan)
            width = child.rows
    fields["plan"] = plan
    fields["calibration"] = _sample(fields["calibration"], rows, i)
    fields["outputs"] = _matrix(fields["outputs"], rows, o)
    fields["activation_inputs"] = tuple(
        _sample(value, rows, width)
        for value, width in zip(
            _array(fields["activation_inputs"], len(widths)),
            widths,
            strict=True,
        )
    )
    fields["matrices"] = tuple(
        joint_matrix_realization_from_dict(matrix, decisions=decisions, limits=limits)
        for matrix in _array(fields["matrices"], len(plan.matrices))
    )
    equation = _record(fields["function"], FrozenMLPFunction)
    if equation.get("version") != "frozen-mlp-function-v1":
        raise ValueError("MLP construction requires its frozen equation version")
    for value, matrix in zip(
        _array(equation["linears"], len(plan.matrices)), plan.matrices, strict=True
    ):
        linear = _record(value, FrozenLinearFunction)
        if linear.get("version") != "frozen-linear-function-v1":
            raise ValueError("MLP construction requires frozen linear maps")
        _matrix(linear["matrix"], matrix.rows, matrix.columns)
    for value, width in zip(
        _array(equation.get("activations", ()), len(widths)), widths, strict=True
    ):
        activation = _record(value, CalibratedActivation)
        if set(activation) != {field.name for field in dataclass_fields(CalibratedActivation)}:
            raise ValueError("MLP fits must retain all activation calibration fields")
        if activation["calibration_version"] != "independent-standard-activation-v1":
            raise ValueError("MLP requires frozen standard activation fits")
        for name in (
            "input_mean",
            "input_inverse_std",
            "shift",
            "output_mean",
            "output_inverse_std",
            "shift_rows",
        ):
            _array(activation[name], width)
    fields["function"] = frozen_function_from_dict(equation, limits=limits)
    result = JointMLPFunctionRealization(**fields)
    result.validate(decisions)
    return result
