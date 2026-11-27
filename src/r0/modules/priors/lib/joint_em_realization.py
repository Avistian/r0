"""Source EM anchor constructions and soft assignments before the lazy linear fit."""

import copy
import pickle
from dataclasses import dataclass

import numpy as np
import torch
from numpy.typing import NDArray

from r0.modules.priors.lib.activation_sampling import finite_number
from r0.modules.priors.lib.causal_limits import CausalLimits
from r0.modules.priors.lib.frozen_function_composites import FrozenEMFunction
from r0.modules.priors.lib.frozen_function_evaluation import em_assignment_probabilities
from r0.modules.priors.lib.frozen_function_primitives import FrozenLinearFunction
from r0.modules.priors.lib.joint_function_admission import admit_em_function
from r0.modules.priors.lib.joint_function_calibration import (
    FunctionCalibrationSample,
    freeze_function_calibration,
)
from r0.modules.priors.lib.joint_function_plans import JointFunctionPlan
from r0.modules.priors.lib.joint_matrix_realization import (
    JointMatrixRealization,
    materialize_joint_matrix,
)
from r0.modules.priors.lib.joint_point_realization import frozen_matrix
from r0.modules.priors.lib.joint_sampling_records import JointSamplingRecord
from r0.modules.priors.lib.weight_geometry import Matrix, matrix_shape


def em_geometry(
    plan: JointFunctionPlan,
    calibration: FunctionCalibrationSample,
    anchor_rows: tuple[int, ...],
    center_gaussian: Matrix,
    spread_gaussian: tuple[float, ...],
) -> tuple[Matrix, tuple[float, ...]]:
    """Derive centers and spreads after validated raw geometry and caller admission."""
    anchors = np.asarray(calibration.sample, dtype=np.float64)[list(anchor_rows)]
    with np.errstate(over="ignore", invalid="ignore", under="ignore"):
        centers = frozen_matrix(anchors + np.asarray(center_gaussian, dtype=np.float64))
        scales = tuple(
            map(
                float,
                np.exp(
                    float(dict(plan.parameters)["spread_factor"])
                    * np.asarray(spread_gaussian, dtype=np.float64)
                ),
            )
        )
    matrix_shape(centers)
    if any(not finite_number(scale) or scale <= 0 for scale in scales):
        raise ValueError("EM spreads must be finite and strictly positive")
    return centers, scales


def em_fit_assignments(
    plan: JointFunctionPlan,
    calibration: FunctionCalibrationSample,
    centers: Matrix,
    scales: tuple[float, ...],
) -> Matrix:
    parameters = dict(plan.parameters)
    values = em_assignment_probabilities(
        torch.tensor(calibration.sample, dtype=torch.float64),
        centers,
        scales,
        float(parameters["distance_p"]),
        float(parameters["distance_q"]),
    )
    return frozen_matrix(values.numpy())


@dataclass(frozen=True, slots=True)
class JointEMFunctionRealization:
    plan: JointFunctionPlan
    calibration: FunctionCalibrationSample
    anchor_rows: tuple[int, ...]
    center_gaussian: Matrix
    spread_gaussian: tuple[float, ...]
    assignments: Matrix
    matrix: JointMatrixRealization
    function: FrozenEMFunction
    version: str = "joint-em-function-realization-v1"

    def __post_init__(self) -> None:
        if (
            self.version != "joint-em-function-realization-v1"
            or type(self.plan) is not JointFunctionPlan
            or self.plan.family != "em"
            or type(self.calibration) is not FunctionCalibrationSample
            or type(self.matrix) is not JointMatrixRealization
            or self.matrix.plan != self.plan.matrices[0]
            or type(self.function) is not FrozenEMFunction
        ):
            raise ValueError("EM realization requires its plan, calibration and owned matrix")
        plan = self.plan
        parameters = dict(plan.parameters)
        count = int(parameters["components"])
        if matrix_shape(self.calibration.sample) != (plan.calibration_rows, plan.input_width):
            raise ValueError("EM calibration differs from its planned geometry")
        if (
            type(self.anchor_rows) is not tuple
            or len(self.anchor_rows) != count
            or any(
                type(index) is not int or not 0 <= index < plan.calibration_rows
                for index in self.anchor_rows
            )
        ):
            raise ValueError("EM anchors require in-range row indices with replacement")
        if (
            matrix_shape(self.center_gaussian) != (count, plan.input_width)
            or type(self.spread_gaussian) is not tuple
            or len(self.spread_gaussian) != count
            or any(not finite_number(value) for value in self.spread_gaussian)
        ):
            raise ValueError("EM Gaussian draws must match planned finite geometry")
        centers, scales = em_geometry(
            plan, self.calibration, self.anchor_rows, self.center_gaussian, self.spread_gaussian
        )
        expected = FrozenEMFunction(
            centers,
            scales,
            FrozenLinearFunction(self.matrix.materialize()[0]),
            float(parameters["distance_p"]),
            float(parameters["distance_q"]),
        )
        if self.function != expected:
            raise ValueError("frozen EM equation differs from its retained fit construction")
        if matrix_shape(self.assignments) != (
            plan.calibration_rows,
            count,
        ) or self.assignments != em_fit_assignments(plan, self.calibration, centers, scales):
            raise ValueError("EM assignments differ from their retained fit construction")

    def validate(self, decisions: JointSamplingRecord) -> None:
        self.plan.validate(decisions)
        self.matrix.validate(decisions)


def materialize_joint_em(
    plan: JointFunctionPlan,
    decisions: JointSamplingRecord,
    *,
    rng: np.random.Generator,
    fit_rng: np.random.Generator | None,
    calibration: NDArray[np.float64] | None,
    calibration_id: str,
    limits: CausalLimits,
) -> JointEMFunctionRealization:
    if type(fit_rng) is not np.random.Generator or fit_rng.bit_generator is rng.bit_generator:
        raise ValueError("EM fitting requires a separate injected fit RNG")
    admit_em_function(plan, decisions, limits)
    sample = freeze_function_calibration(plan, calibration, calibration_id)
    raw_before, fit_before = (
        pickle.dumps(rng.bit_generator.state),
        pickle.dumps(fit_rng.bit_generator.state),
    )
    raw, fit = copy.deepcopy(rng), copy.deepcopy(fit_rng)
    count = int(dict(plan.parameters)["components"])
    rows = tuple(map(int, fit.integers(plan.calibration_rows, size=count)))
    jitter = frozen_matrix(raw.normal(size=(count, plan.input_width)))
    spread = tuple(map(float, raw.normal(size=count)))
    centers, scales = em_geometry(plan, sample, rows, jitter, spread)
    assignments = em_fit_assignments(plan, sample, centers, scales)
    # The source output linear is lazy: realize it only after assignment evaluation.
    matrix = materialize_joint_matrix(plan.matrices[0], decisions, rng=raw, limits=limits)
    parameters = dict(plan.parameters)
    function = FrozenEMFunction(
        centers,
        scales,
        FrozenLinearFunction(matrix.materialize()[0]),
        float(parameters["distance_p"]),
        float(parameters["distance_q"]),
    )
    result = JointEMFunctionRealization(
        plan, sample, rows, jitter, spread, assignments, matrix, function
    )
    result.validate(decisions)
    if (
        pickle.dumps(rng.bit_generator.state) != raw_before
        or pickle.dumps(fit_rng.bit_generator.state) != fit_before
    ):
        raise ValueError("caller RNG changed during EM construction")
    rng.bit_generator.state = raw.bit_generator.state
    fit_rng.bit_generator.state = fit.bit_generator.state
    return result
