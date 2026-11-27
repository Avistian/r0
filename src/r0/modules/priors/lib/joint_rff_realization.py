"""Raw source Fourier draws and their fixed, reconstructible numerical equation."""

import copy
from dataclasses import dataclass

import numpy as np

from r0.modules.priors.lib.activation_sampling import finite_number
from r0.modules.priors.lib.causal_limits import CausalLimits
from r0.modules.priors.lib.frozen_function_primitives import FrozenRFFFunction
from r0.modules.priors.lib.joint_function_admission import admit_rff_function
from r0.modules.priors.lib.joint_function_plans import JointFunctionPlan
from r0.modules.priors.lib.joint_matrix_realization import bind_weight_batch
from r0.modules.priors.lib.joint_point_realization import frozen_matrix
from r0.modules.priors.lib.joint_sampling_records import JointSamplingRecord
from r0.modules.priors.lib.joint_weight_plans import materialize_joint_weights
from r0.modules.priors.lib.weight_geometry import Matrix, matrix_shape
from r0.modules.priors.lib.weight_vectors import WeightVector


def rff_equation(
    plan: JointFunctionPlan,
    uniform: Matrix,
    phase: tuple[float, ...],
    coefficients: Matrix,
    directions: Matrix | None = None,
    factor: Matrix | None = None,
    weights: WeightVector | None = None,
) -> FrozenRFFFunction:
    a = dict(plan.parameters)["a"]
    values = np.asarray(uniform, dtype=np.float64)
    frequencies = np.clip(values, 1e-6, 1 - 1e-6) ** (1 / (1 - a)) - 1
    if dict(plan.parameters)["product_kernel"] is False:
        assert directions is not None
        assert factor is not None
        assert weights is not None
        normals = np.asarray(directions, dtype=np.float64)
        magnitude = np.abs(normals).max(axis=0)
        if np.any(magnitude == 0):
            raise ValueError("zero Fourier direction; construction was not retried")
        scaled = normals / magnitude
        unit = scaled / np.sqrt(np.square(scaled).sum(axis=0))
        gaussian = np.asarray(factor, dtype=np.float64)
        vector = np.asarray(weights.values(), dtype=np.float64)
        with np.errstate(over="ignore", invalid="ignore"):
            transform = gaussian * vector[:, None] * dict(plan.parameters)["scale"]
            frequencies = transform @ (unit * frequencies)
        if not np.isfinite(frequencies).all():
            raise ValueError("nonfinite Fourier construction; task was not retried")
    phases = tuple(float(2 * np.pi * p) for p in phase)
    scaled_coefficients = np.asarray(coefficients, dtype=np.float64) / np.sqrt(
        plan.prior.frequencies
    )
    return FrozenRFFFunction(frozen_matrix(frequencies), phases, frozen_matrix(scaled_coefficients))


@dataclass(frozen=True, slots=True)
class JointRFFFunctionRealization:
    plan: JointFunctionPlan
    uniform: Matrix
    phase_uniform: tuple[float, ...]
    coefficient_gaussian: Matrix
    function: FrozenRFFFunction
    directions: Matrix | None = None
    factor: Matrix | None = None
    weights: WeightVector | None = None
    version: str = "joint-rff-function-realization-v1"

    def __post_init__(self) -> None:
        if (
            self.version != "joint-rff-function-realization-v1"
            or type(self.plan) is not JointFunctionPlan
            or self.plan.family != "gp"
            or type(self.function) is not FrozenRFFFunction
        ):
            raise ValueError(
                "Fourier realization requires its version, GP plan and frozen equation"
            )
        plan = self.plan
        i, o, k = plan.input_width, plan.output_width, plan.prior.frequencies
        product = dict(plan.parameters)["product_kernel"] is True
        if (
            matrix_shape(self.uniform) != (i if product else 1, k)
            or any(not 0 <= u < 1 for row in self.uniform for u in row)
            or type(self.phase_uniform) is not tuple
            or len(self.phase_uniform) != k
            or any(not finite_number(u) or not 0 <= u < 1 for u in self.phase_uniform)
            or matrix_shape(self.coefficient_gaussian) != (k, o)
        ):
            raise ValueError(
                "Fourier raw draws differ from their source geometry or uniform bounds"
            )
        if product:
            if self.directions is not None or self.factor is not None or self.weights is not None:
                raise ValueError("product GP cannot own radial factor fields")
        else:
            if (
                self.directions is None
                or matrix_shape(self.directions) != (i, k)
                or self.factor is None
                or matrix_shape(self.factor) != (i, i)
                or self.weights is None
            ):
                raise ValueError("radial GP requires all raw direction and factor fields")
            assert plan.weights is not None
            bind_weight_batch(plan.weights, (self.weights,))
        expected = rff_equation(
            plan,
            self.uniform,
            self.phase_uniform,
            self.coefficient_gaussian,
            self.directions,
            self.factor,
            self.weights,
        )
        if self.function != expected:
            raise ValueError("frozen Fourier equation differs from its retained construction")

    def validate(self, decisions: JointSamplingRecord) -> None:
        self.plan.validate(decisions)


def materialize_joint_rff(
    plan: JointFunctionPlan,
    decisions: JointSamplingRecord,
    *,
    rng: np.random.Generator,
    limits: CausalLimits,
) -> JointRFFFunctionRealization:
    admit_rff_function(plan, decisions, limits)
    staged = copy.deepcopy(rng)
    i, o, k = plan.input_width, plan.output_width, plan.prior.frequencies
    product = dict(plan.parameters)["product_kernel"] is True
    uniform = frozen_matrix(staged.random((i if product else 1, k)))
    directions, factor, weights = None, None, None
    if not product:
        directions = frozen_matrix(staged.normal(size=(i, k)))
        factor = frozen_matrix(staged.normal(size=(i, i)))
        assert plan.weights is not None
        weights = materialize_joint_weights(plan.weights, decisions, rng=staged, limits=limits)[0]
    phase = tuple(float(u) for u in staged.random(k))
    coefficients = frozen_matrix(staged.normal(size=(k, o)))
    function = rff_equation(plan, uniform, phase, coefficients, directions, factor, weights)
    result = JointRFFFunctionRealization(
        plan, uniform, phase, coefficients, function, directions, factor, weights
    )
    result.validate(decisions)
    rng.bit_generator.state = staged.bit_generator.state
    return result
