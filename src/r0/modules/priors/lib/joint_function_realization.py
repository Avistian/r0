"""Atomic raw matrix function constructions bound to their frozen equations."""

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from r0.modules.priors.lib.causal_limits import CausalLimits
from r0.modules.priors.lib.frozen_function_composites import FrozenProductFunction
from r0.modules.priors.lib.frozen_function_primitives import (
    FrozenLinearFunction,
    FrozenQuadraticFunction,
)
from r0.modules.priors.lib.joint_discretization_realization import (
    JointDiscretizationFunctionRealization,
    materialize_joint_discretization,
)
from r0.modules.priors.lib.joint_em_realization import (
    JointEMFunctionRealization,
    materialize_joint_em,
)
from r0.modules.priors.lib.joint_function_admission import (
    admit_matrix_function,
    admit_product_function,
)
from r0.modules.priors.lib.joint_function_calibration import (
    FunctionCalibrationSample,
    freeze_function_calibration,
)
from r0.modules.priors.lib.joint_function_plans import JointFunctionPlan
from r0.modules.priors.lib.joint_matrix_realization import (
    JointMatrixRealization,
    materialize_joint_matrix,
)
from r0.modules.priors.lib.joint_mlp_realization import (
    JointMLPFunctionRealization,
    materialize_joint_mlp,
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
from r0.modules.priors.lib.raw_draw_capture import (
    RawDrawLedger,
    raw_transaction,
    raw_transactions,
    recorded_draw,
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


type CheapJointFunctionRealization = (
    JointMatrixFunctionRealization
    | JointRFFFunctionRealization
    | JointTreeFunctionRealization
    | JointDiscretizationFunctionRealization
)

_CHEAP_REALIZATION_TYPES = (
    JointMatrixFunctionRealization,
    JointRFFFunctionRealization,
    JointTreeFunctionRealization,
    JointDiscretizationFunctionRealization,
)


@dataclass(frozen=True, slots=True)
class JointProductFunctionRealization:
    plan: JointFunctionPlan
    left: CheapJointFunctionRealization
    right: CheapJointFunctionRealization
    function: FrozenProductFunction
    calibration: FunctionCalibrationSample | None = None
    version: str = "joint-product-function-realization-v1"

    def __post_init__(self) -> None:
        if (
            self.version != "joint-product-function-realization-v1"
            or type(self.plan) is not JointFunctionPlan
            or self.plan.family != "prod"
            or type(self.left) not in _CHEAP_REALIZATION_TYPES
            or type(self.right) not in _CHEAP_REALIZATION_TYPES
            or type(self.function) is not FrozenProductFunction
        ):
            raise ValueError("product realization requires its plan and two cheap children")
        if (self.left.plan, self.right.plan) != self.plan.children:
            raise ValueError("product children differ from their ordered owned plans")
        fitted = tuple(
            child
            for child in (self.left, self.right)
            if isinstance(
                child, (JointTreeFunctionRealization, JointDiscretizationFunctionRealization)
            )
        )
        if fitted:
            if type(self.calibration) is not FunctionCalibrationSample or any(
                child.calibration != self.calibration for child in fitted
            ):
                raise ValueError("product fitted children must share the same retained calibration")
        elif self.calibration is not None:
            raise ValueError("product of unfitted functions cannot own a calibration sample")
        if self.function != FrozenProductFunction(self.left.function, self.right.function):
            raise ValueError("frozen product differs from its retained child constructions")

    def validate(self, decisions: JointSamplingRecord) -> None:
        self.plan.validate(decisions)
        self.left.validate(decisions)
        self.right.validate(decisions)


def materialize_joint_function(
    plan: JointFunctionPlan,
    decisions: JointSamplingRecord,
    *,
    rng: np.random.Generator,
    fit_rng: np.random.Generator | None = None,
    calibration: NDArray[np.float64] | None = None,
    calibration_id: str = "",
    limits: CausalLimits = _DEFAULT_LIMITS,
    capture: RawDrawLedger | None = None,
) -> (
    JointMatrixFunctionRealization
    | JointRFFFunctionRealization
    | JointTreeFunctionRealization
    | JointDiscretizationFunctionRealization
    | JointEMFunctionRealization
    | JointProductFunctionRealization
    | JointMLPFunctionRealization
):
    """Freeze supported constructions using caller-owned samples for fitted functions.

    The graph fitter must establish sample independence and ancestral provenance.
    This helper binds retained numerical inputs and stages its injected RNGs.
    """
    if type(plan) is not JointFunctionPlan or type(decisions) is not JointSamplingRecord:
        raise ValueError("function realization requires a frozen plan and decisions")
    if type(rng) is not np.random.Generator or type(limits) is not CausalLimits:
        raise ValueError("function realization requires an injected RNG and frozen limits")
    plan.validate(decisions)
    if plan.family == "mlp":
        return materialize_joint_mlp(
            plan,
            decisions,
            rng=rng,
            fit_rng=fit_rng,
            calibration=calibration,
            calibration_id=calibration_id,
            limits=limits,
            capture=capture,
        )
    if plan.family == "prod":
        return _materialize_product(
            plan,
            decisions,
            rng=rng,
            fit_rng=fit_rng,
            calibration=calibration,
            calibration_id=calibration_id,
            limits=limits,
            capture=capture,
        )
    if plan.family == "em":
        return materialize_joint_em(
            plan,
            decisions,
            rng=rng,
            fit_rng=fit_rng,
            calibration=calibration,
            calibration_id=calibration_id,
            limits=limits,
            capture=capture,
        )
    if plan.family == "disc":
        return materialize_joint_discretization(
            plan,
            decisions,
            rng=rng,
            fit_rng=fit_rng,
            calibration=calibration,
            calibration_id=calibration_id,
            limits=limits,
            capture=capture,
        )
    if plan.family == "tree":
        return materialize_joint_tree(
            plan,
            decisions,
            rng=rng,
            fit_rng=fit_rng,
            calibration=calibration,
            calibration_id=calibration_id,
            limits=limits,
            capture=capture,
        )
    if plan.family == "gp":
        return materialize_joint_rff(plan, decisions, rng=rng, limits=limits, capture=capture)
    admit_matrix_function(plan, decisions, limits)
    with raw_transaction(rng, capture) as (staged, capture):
        permutation = (
            tuple(
                int(i)
                for i in recorded_draw(
                    staged,
                    capture,
                    owner=plan.operation,
                    role="quadratic-permutation",
                    distribution="permutation",
                    shape=(plan.input_width,),
                    parameters=(("count", plan.input_width),),
                )
            )
            if plan.family == "quad" and plan.input_width > 20
            else ()
        )
        matrix = materialize_joint_matrix(
            plan.matrices[0], decisions, rng=staged, limits=limits, capture=capture
        )
        matrices = matrix.materialize()
        function: FrozenLinearFunction | FrozenQuadraticFunction
        if plan.family == "lin":
            function = FrozenLinearFunction(matrices[0])
        else:
            indices = permutation[:20] if permutation else tuple(range(plan.input_width))
            function = FrozenQuadraticFunction(plan.input_width, indices, matrices)
        result = JointMatrixFunctionRealization(plan, matrix, function, permutation)
        result.validate(decisions)
        return result


def _materialize_product(
    plan: JointFunctionPlan,
    decisions: JointSamplingRecord,
    *,
    rng: np.random.Generator,
    fit_rng: np.random.Generator | None,
    calibration: NDArray[np.float64] | None,
    calibration_id: str,
    limits: CausalLimits,
    capture: RawDrawLedger | None = None,
) -> JointProductFunctionRealization:
    fitted = any(child.family in ("tree", "disc") for child in plan.children)
    if (fitted and fit_rng is None) or (
        fit_rng is not None
        and (type(fit_rng) is not np.random.Generator or fit_rng.bit_generator is rng.bit_generator)
    ):
        raise ValueError("product fitting requires a separate injected fit RNG")
    admit_product_function(plan, decisions, limits)
    sample = freeze_function_calibration(plan, calibration, calibration_id) if fitted else None
    shared = np.asarray(sample.sample, dtype=np.float64) if sample is not None else None
    with raw_transactions(
        {"raw": rng, **({"fit": fit_rng} if fit_rng is not None else {})}, capture
    ) as (streams, capture):
        raw, fit = streams["raw"], streams.get("fit")
        children: list[CheapJointFunctionRealization] = []
        for child_plan in plan.children:
            child = materialize_joint_function(
                child_plan,
                decisions,
                rng=raw,
                fit_rng=fit,
                calibration=shared,
                calibration_id=calibration_id,
                limits=limits,
                capture=capture,
            )
            if not isinstance(
                child,
                (
                    JointMatrixFunctionRealization,
                    JointRFFFunctionRealization,
                    JointTreeFunctionRealization,
                    JointDiscretizationFunctionRealization,
                ),
            ):
                raise ValueError("product construction requires cheap child realizations")
            children.append(child)
        left, right = children
        result = JointProductFunctionRealization(
            plan,
            left,
            right,
            FrozenProductFunction(left.function, right.function),
            sample,
        )
        result.validate(decisions)
        return result
