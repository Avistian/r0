"""Strict branch-shaped reconstruction of retained Fourier constructions."""

from dataclasses import fields as dataclass_fields
from typing import Any, cast

from r0.modules.priors.lib.causal_limits import CausalLimits
from r0.modules.priors.lib.frozen_function_decoding import frozen_function_from_dict
from r0.modules.priors.lib.frozen_function_primitives import FrozenRFFFunction
from r0.modules.priors.lib.joint_function_admission import admit_rff_function
from r0.modules.priors.lib.joint_function_decoding import joint_function_plan_from_dict
from r0.modules.priors.lib.joint_rff_realization import JointRFFFunctionRealization
from r0.modules.priors.lib.joint_sampling_records import JointSamplingRecord
from r0.modules.priors.lib.weight_geometry import Matrix
from r0.modules.priors.lib.weight_vectors import (
    PolynomialWeightVector,
    UniformWeightVector,
    weight_vector_from_dict,
)


def _record(value: Any, record_type: type[Any]) -> dict[str, Any]:
    if type(value) is not dict:
        raise ValueError("Fourier records must be JSON objects")
    result = cast(dict[str, Any], value)
    if set(result) - {field.name for field in dataclass_fields(record_type)}:
        raise ValueError("Fourier record contains unknown fields")
    return dict(result)


def _array(value: Any, count: int) -> tuple[Any, ...]:
    if type(value) not in (list, tuple) or len(value) != count:
        raise ValueError("Fourier array must match its exact planned geometry")
    return tuple(value)


def _matrix(value: Any, rows: int, columns: int) -> Matrix:
    return tuple(_array(row, columns) for row in _array(value, rows))


def decode_joint_rff(
    data: dict[str, Any],
    decisions: JointSamplingRecord,
    limits: CausalLimits,
) -> JointRFFFunctionRealization:
    """The public enclosing decoder admits the complete raw payload before this call."""
    fields = _record(data, JointRFFFunctionRealization)
    plan = joint_function_plan_from_dict(fields["plan"], decisions=decisions, limits=limits)
    admit_rff_function(plan, decisions, limits)
    i, o, k = plan.input_width, plan.output_width, plan.prior.frequencies
    product = dict(plan.parameters)["product_kernel"] is True
    fields["plan"] = plan
    fields["uniform"] = _matrix(fields["uniform"], i if product else 1, k)
    fields["phase_uniform"] = _array(fields["phase_uniform"], k)
    fields["coefficient_gaussian"] = _matrix(fields["coefficient_gaussian"], k, o)
    if product:
        if any(fields.get(name) is not None for name in ("directions", "factor", "weights")):
            raise ValueError("product GP cannot own radial factor fields")
    else:
        fields["directions"] = _matrix(fields["directions"], i, k)
        fields["factor"] = _matrix(fields["factor"], i, i)
        assert plan.weights is not None
        kind = UniformWeightVector if plan.weights.family == "uniform" else PolynomialWeightVector
        weights = _record(fields["weights"], kind)
        if weights.get("version") != f"{plan.weights.family}-weight-vector-v1":
            raise ValueError("Fourier factor weight version differs from its plan")
        weights["draws"] = _array(weights["draws"], i)
        if kind is PolynomialWeightVector:
            weights["permutation"] = _array(weights["permutation"], i)
        fields["weights"] = weight_vector_from_dict(weights)
    equation = _record(fields["function"], FrozenRFFFunction)
    if equation.get("version") != "frozen-rff-function-v1":
        raise ValueError("Fourier construction requires a frozen Fourier equation")
    for name, rows, columns in (("frequencies", i, k), ("coefficients", k, o)):
        _matrix(equation[name], rows, columns)
    _array(equation["phases"], k)
    fields["function"] = frozen_function_from_dict(equation, limits=limits)
    result = JointRFFFunctionRealization(**fields)
    result.validate(decisions)
    return result
