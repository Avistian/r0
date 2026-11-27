"""Atomic raw matrix function constructions bound to their frozen equations."""

import copy
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from r0.modules.priors.lib.causal_limits import CausalLimits
from r0.modules.priors.lib.frozen_function_primitives import (
    FrozenLinearFunction,
    FrozenQuadraticFunction,
)
from r0.modules.priors.lib.joint_function_admission import admit_matrix_function
from r0.modules.priors.lib.joint_function_plans import JointFunctionPlan
from r0.modules.priors.lib.joint_matrix_realization import (
    JointMatrixRealization,
    materialize_joint_matrix,
)
from r0.modules.priors.lib.joint_rff_realization import (
    JointRFFFunctionRealization,
    materialize_joint_rff,
)
from r0.modules.priors.lib.joint_sampling_records import JointSamplingRecord
from r0.modules.priors.lib.joint_tree_realization import (
    JointTreeFunctionRealization,
    materialize_joint_tree,
)

_DEFAULT_LIMITS = CausalLimits()


@dataclass(frozen=True, slots=True)
class JointMatrixFunctionRealization:
    plan: JointFunctionPlan
    matrix: JointMatrixRealization
    function: FrozenLinearFunction | FrozenQuadraticFunction
    permutation: tuple[int, ...] = ()
    version: str = "joint-matrix-function-realization-v1"

    def __post_init__(self) -> None:
        if (
            self.version != "joint-matrix-function-realization-v1"
            or type(self.plan) is not JointFunctionPlan
            or self.plan.family not in ("lin", "quad")
            or type(self.matrix) is not JointMatrixRealization
            or self.matrix.plan != self.plan.matrices[0]
            or type(self.permutation) is not tuple
        ):
            raise ValueError(
                "matrix function realization requires its frozen plan and owned matrix"
            )
        matrices = self.matrix.materialize()
        expected: FrozenLinearFunction | FrozenQuadraticFunction
        if self.plan.family == "lin":
            if self.permutation != () or type(self.function) is not FrozenLinearFunction:
                raise ValueError("linear realization cannot own a feature permutation")
            expected = FrozenLinearFunction(matrices[0])
        else:
            width = self.plan.input_width
            if type(self.function) is not FrozenQuadraticFunction:
                raise ValueError("quadratic realization requires a frozen quadratic equation")
            if width > 20:
                if (
                    len(self.permutation) != width
                    or any(type(i) is not int or not 0 <= i < width for i in self.permutation)
                    or len(set(self.permutation)) != width
                ):
                    raise ValueError(
                        "quadratic realization must retain its complete raw permutation"
                    )
                indices = self.permutation[:20]
            else:
                if self.permutation != ():
                    raise ValueError("small quadratic realization has deterministic feature order")
                indices = tuple(range(width))
            expected = FrozenQuadraticFunction(width, indices, matrices)
        if self.function != expected:
            raise ValueError("frozen function equation differs from its retained construction")

    def validate(self, decisions: JointSamplingRecord) -> None:
        self.plan.validate(decisions)
        self.matrix.validate(decisions)


def materialize_joint_function(
    plan: JointFunctionPlan,
    decisions: JointSamplingRecord,
    *,
    rng: np.random.Generator,
    fit_rng: np.random.Generator | None = None,
    calibration: NDArray[np.float64] | None = None,
    calibration_id: str = "",
    limits: CausalLimits = _DEFAULT_LIMITS,
) -> JointMatrixFunctionRealization | JointRFFFunctionRealization | JointTreeFunctionRealization:
    """Freeze supported constructions; tree fits use caller-owned calibration samples.

    The graph fitter must establish sample independence and ancestral provenance.
    This helper binds retained numerical inputs and stages its injected RNGs.
    """
    if type(plan) is not JointFunctionPlan or type(decisions) is not JointSamplingRecord:
        raise ValueError("function realization requires a frozen plan and decisions")
    if type(rng) is not np.random.Generator or type(limits) is not CausalLimits:
        raise ValueError("function realization requires an injected RNG and frozen limits")
    plan.validate(decisions)
    if plan.family == "tree":
        return materialize_joint_tree(
            plan,
            decisions,
            rng=rng,
            fit_rng=fit_rng,
            calibration=calibration,
            calibration_id=calibration_id,
            limits=limits,
        )
    if plan.family == "gp":
        return materialize_joint_rff(plan, decisions, rng=rng, limits=limits)
    admit_matrix_function(plan, decisions, limits)
    staged = copy.deepcopy(rng)
    permutation = (
        tuple(int(i) for i in staged.permutation(plan.input_width))
        if plan.family == "quad" and plan.input_width > 20
        else ()
    )
    matrix = materialize_joint_matrix(plan.matrices[0], decisions, rng=staged, limits=limits)
    matrices = matrix.materialize()
    function: FrozenLinearFunction | FrozenQuadraticFunction
    if plan.family == "lin":
        function = FrozenLinearFunction(matrices[0])
    else:
        indices = permutation[:20] if permutation else tuple(range(plan.input_width))
        function = FrozenQuadraticFunction(plan.input_width, indices, matrices)
    result = JointMatrixFunctionRealization(plan, matrix, function, permutation)
    result.validate(decisions)
    rng.bit_generator.state = staged.bit_generator.state
    return result
