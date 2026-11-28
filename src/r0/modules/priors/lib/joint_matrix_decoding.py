"""Bound raw matrix geometry before constructing any retained array tuples."""

from dataclasses import fields as dataclass_fields
from typing import Any, cast

from r0.modules.priors.lib.causal_activations import CausalActivation
from r0.modules.priors.lib.causal_limits import CausalLimits
from r0.modules.priors.lib.joint_matrix_plans import JointMatrixPlan, joint_matrix_plan_from_dict
from r0.modules.priors.lib.joint_matrix_realization import (
    JointMatrixRealization,
    admit_matrix_realization,
)
from r0.modules.priors.lib.joint_sampling_records import JointSamplingRecord
from r0.modules.priors.lib.joint_weight_plans import JointWeightPlan
from r0.modules.priors.lib.kernel_points import KernelPointCloud, point_cloud_from_dict
from r0.modules.priors.lib.matrix_preference_prior import FAMILY_VERSIONS
from r0.modules.priors.lib.transformed_weights import (
    GaussianWeights,
    NormalizedWeights,
    TransformedWeights,
)
from r0.modules.priors.lib.weight_constructions import weight_construction_from_dict
from r0.modules.priors.lib.weight_matrices import (
    ImportanceFactorizedWeights,
    ImportanceWeights,
    LaplaceKernelWeights,
)
from r0.modules.priors.lib.weight_vectors import PolynomialWeightVector, UniformWeightVector

_DEFAULT_LIMITS = CausalLimits()


def _raw_budget(data: Any, limits: CausalLimits) -> None:
    # This controls JSON traversal overhead, separately from the exact geometry
    # admission below. Counters include metadata and containers, not just floats.
    remaining = 32 * limits.parameters + 1024

    def visit(value: Any, depth: int) -> None:
        nonlocal remaining
        remaining -= 1
        if remaining < 0 or depth > 24:
            raise ValueError("raw matrix payload exceeds its traversal limit")
        if type(value) is dict:
            mapping = cast(dict[Any, Any], value)
            if len(mapping) > remaining:
                raise ValueError("raw matrix object exceeds its traversal limit")
            for key, item in mapping.items():
                if type(key) is not str or len(key.encode("utf-8")) > 128:
                    raise ValueError("matrix JSON keys must be bounded strings")
                visit(item, depth + 1)
        elif type(value) in (list, tuple):
            sequence = cast(list[Any] | tuple[Any, ...], value)
            if len(sequence) > remaining:
                raise ValueError("raw matrix array exceeds its traversal limit")
            for item in sequence:
                visit(item, depth + 1)
        elif type(value) is str:
            if len(value.encode("utf-8")) > 128:
                raise ValueError("matrix JSON strings must be bounded")
        elif value is not None and type(value) not in (bool, int, float):
            raise ValueError("matrix payload must contain JSON primitives")

    visit(data, 0)


def _array(value: Any, length: int) -> list[Any] | tuple[Any, ...]:
    if type(value) not in (list, tuple) or len(value) != length:
        raise ValueError("matrix array must match its planned geometry")
    return value


def _matrix(value: Any, rows: int, columns: int) -> None:
    for row in _array(value, rows):
        _array(row, columns)


def _object(value: Any) -> dict[str, Any]:
    if type(value) is not dict:
        raise ValueError("matrix record must be a JSON object")
    return cast(dict[str, Any], value)


def _record(value: Any, record_type: type[Any]) -> dict[str, Any]:
    fields = _object(value)
    if set(fields) - {f.name for f in dataclass_fields(record_type)}:
        raise ValueError("matrix record has unknown fields")
    return fields


def _vector(value: Any, plan: JointWeightPlan) -> None:
    kind = UniformWeightVector if plan.family == "uniform" else PolynomialWeightVector
    fields = _record(value, kind)
    if fields.get("version") != f"{plan.family}-weight-vector-v1":
        raise ValueError("weight realization version differs from its planned family")
    _array(fields["draws"], plan.width)
    if plan.family == "polynomial":
        _array(fields["permutation"], plan.width)


def _member(value: Any, plan: JointMatrixPlan) -> None:
    fields = _record(value, NormalizedWeights)
    if fields.get("version") != "normalized-weights-v1":
        raise ValueError("joint matrix member requires its source normalization")
    n, m = plan.rows, plan.columns
    _matrix(fields["jitter"], n, m)
    kinds = {
        "gaussian": GaussianWeights,
        "importance": ImportanceWeights,
        "factorized": ImportanceFactorizedWeights,
        "kernel": LaplaceKernelWeights,
        "transformed": TransformedWeights,
    }
    source = _record(fields["source"], kinds[plan.family])
    if source.get("version") != FAMILY_VERSIONS[plan.family]:
        raise ValueError("matrix construction version differs from its planned family")
    if plan.family == "gaussian":
        _matrix(source["draws"], n, m)
    elif plan.family == "importance":
        assert plan.weights is not None
        _matrix(source["gaussian"], n, m)
        for vector in _array(source["importance"], n):
            _vector(vector, plan.weights)
    elif plan.family == "factorized":
        assert plan.weights is not None
        k = min(n, m)
        _matrix(source["left"], n, k)
        _matrix(source["right"], k, m)
        _vector(source["diagonal"], plan.weights)
    elif plan.family == "kernel":
        _matrix(source["left_points"], n, 3)
        _matrix(source["right_points"], m, 3)
        _matrix(source["signs"], n, m)
        if source.get("point_construction") is not None:
            raise ValueError("joint kernel members must reference the one shared cloud")
    else:
        assert plan.child is not None
        _member(source["child"], plan.child)
        _matrix(source["jitter"], n, m)
        activation = _record(source["activation"], CausalActivation)
        parameters = activation.get("parameters", ())
        if type(parameters) not in (list, tuple) or len(parameters) > 1:
            raise ValueError("matrix activation parameters must be a bounded array")
        for pair in parameters:
            _array(pair, 2)


def _cloud(value: Any, plan: JointMatrixPlan) -> None:
    point_plan = plan.points if plan.child is None else plan.child.points
    if point_plan is None:
        if value is not None:
            raise ValueError("non-kernel matrix cannot retain a shared cloud")
        return
    fields = _record(value, KernelPointCloud)
    c, d, q = point_plan.components, point_plan.dimension, point_plan.count
    _matrix(fields["centers"], c, d)
    _matrix(fields["innovations"], q, d)
    for factor in _array(fields["factors"], c):
        _matrix(factor, d, d)
    for vector in _array(fields["factor_weights"], c):
        _vector(vector, point_plan.factor_weights)
    _vector(fields["component_weights"], point_plan.mixture_weights)
    _array(fields["assignments"], q)
    _array(fields["scale_draws"], c)


def joint_matrix_realization_from_dict(
    data: dict[str, Any],
    *,
    decisions: JointSamplingRecord,
    limits: CausalLimits = _DEFAULT_LIMITS,
) -> JointMatrixRealization:
    if type(limits) is not CausalLimits:
        raise ValueError("matrix decoding requires caller-owned frozen limits")
    _raw_budget(data, limits)
    fields = dict(_record(data, JointMatrixRealization))
    plan = joint_matrix_plan_from_dict(fields["plan"], decisions=decisions)
    admit_matrix_realization(plan, limits)
    members = _array(fields["matrices"], plan.batch)
    for member in members:
        _member(member, plan)
    _cloud(fields.get("point_cloud"), plan)
    matrices: list[NormalizedWeights] = []
    for member in members:
        matrix = weight_construction_from_dict(member)
        if type(matrix) is not NormalizedWeights:
            raise ValueError("joint matrix member must be normalized")
        matrices.append(matrix)
    fields["plan"] = plan
    fields["matrices"] = tuple(matrices)
    if fields.get("point_cloud") is not None:
        fields["point_cloud"] = point_cloud_from_dict(fields["point_cloud"])
    result = JointMatrixRealization(**fields)
    result.validate(decisions)
    result.materialize()
    return result
