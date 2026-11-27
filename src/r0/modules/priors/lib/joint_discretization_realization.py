"""Retained calibration selection and linear targets for source discretization."""

import copy
import pickle
from dataclasses import dataclass

import numpy as np
import torch
from numpy.typing import NDArray

from r0.modules.priors.lib.causal_limits import CausalLimits
from r0.modules.priors.lib.frozen_function_composites import FrozenDiscretizationFunction
from r0.modules.priors.lib.frozen_function_primitives import FrozenLinearFunction
from r0.modules.priors.lib.joint_function_admission import admit_discretization_function
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


def discretization_targets(function: FrozenDiscretizationFunction) -> Matrix:
    """Freeze the source center-map multiplication in CPU float64 arithmetic."""
    centers = torch.tensor(function.centers, dtype=torch.float64)
    matrix = torch.tensor(function.linear.matrix, dtype=torch.float64)
    result = frozen_matrix((centers @ matrix.T).numpy())
    matrix_shape(result)
    return result


@dataclass(frozen=True, slots=True)
class JointDiscretizationFunctionRealization:
    plan: JointFunctionPlan
    calibration: FunctionCalibrationSample
    permutation: tuple[int, ...]
    matrix: JointMatrixRealization
    targets: Matrix
    function: FrozenDiscretizationFunction
    version: str = "joint-discretization-function-realization-v1"

    def __post_init__(self) -> None:
        if (
            self.version != "joint-discretization-function-realization-v1"
            or type(self.plan) is not JointFunctionPlan
            or self.plan.family != "disc"
            or type(self.calibration) is not FunctionCalibrationSample
            or type(self.matrix) is not JointMatrixRealization
            or self.matrix.plan != self.plan.matrices[0]
            or type(self.function) is not FrozenDiscretizationFunction
        ):
            raise ValueError(
                "discretization realization requires its plan, calibration and owned matrix"
            )
        plan = self.plan
        rows = plan.calibration_rows
        if matrix_shape(self.calibration.sample) != (rows, plan.input_width):
            raise ValueError("discretization calibration differs from its planned geometry")
        count = plan.center_count
        assert count is not None
        size = rows if rows > 1 else 0
        if (
            type(self.permutation) is not tuple
            or len(self.permutation) != size
            or any(type(index) is not int or not 0 <= index < rows for index in self.permutation)
            or len(set(self.permutation)) != size
        ):
            raise ValueError(
                "discretization requires the full row permutation, or none for one row"
            )
        selected = self.permutation[:count] if rows > 1 else (0,)
        centers = tuple(self.calibration.sample[index] for index in selected)
        expected = FrozenDiscretizationFunction(
            centers,
            FrozenLinearFunction(self.matrix.materialize()[0]),
            float(dict(plan.parameters)["distance_p"]),
        )
        if self.function != expected:
            raise ValueError("frozen discretization differs from its retained fit construction")
        if matrix_shape(self.targets) != (
            count,
            plan.output_width,
        ) or self.targets != discretization_targets(expected):
            raise ValueError(
                "discretization targets differ from the retained center-map construction"
            )

    def validate(self, decisions: JointSamplingRecord) -> None:
        self.plan.validate(decisions)
        self.matrix.validate(decisions)


def materialize_joint_discretization(
    plan: JointFunctionPlan,
    decisions: JointSamplingRecord,
    *,
    rng: np.random.Generator,
    fit_rng: np.random.Generator | None,
    calibration: NDArray[np.float64] | None,
    calibration_id: str,
    limits: CausalLimits,
) -> JointDiscretizationFunctionRealization:
    if type(fit_rng) is not np.random.Generator or fit_rng.bit_generator is rng.bit_generator:
        raise ValueError("discretization fitting requires a separate injected fit RNG")
    admit_discretization_function(plan, decisions, limits)
    sample = freeze_function_calibration(plan, calibration, calibration_id)
    raw_before, fit_before = (
        pickle.dumps(rng.bit_generator.state),
        pickle.dumps(fit_rng.bit_generator.state),
    )
    raw, fit = copy.deepcopy(rng), copy.deepcopy(fit_rng)
    permutation = (
        tuple(map(int, fit.permutation(plan.calibration_rows))) if plan.calibration_rows > 1 else ()
    )
    selected = permutation[: plan.center_count] if plan.calibration_rows > 1 else (0,)
    centers = tuple(sample.sample[index] for index in selected)
    matrix = materialize_joint_matrix(plan.matrices[0], decisions, rng=raw, limits=limits)
    function = FrozenDiscretizationFunction(
        centers,
        FrozenLinearFunction(matrix.materialize()[0]),
        float(dict(plan.parameters)["distance_p"]),
    )
    result = JointDiscretizationFunctionRealization(
        plan,
        sample,
        permutation,
        matrix,
        discretization_targets(function),
        function,
    )
    result.validate(decisions)
    if (
        pickle.dumps(rng.bit_generator.state) != raw_before
        or pickle.dumps(fit_rng.bit_generator.state) != fit_before
    ):
        raise ValueError("caller RNG changed during discretization construction")
    rng.bit_generator.state = raw.bit_generator.state
    fit_rng.bit_generator.state = fit.bit_generator.state
    return result
