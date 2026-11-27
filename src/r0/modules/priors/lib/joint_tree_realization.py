"""Source oblivious-tree fits with retained calibration, selections and leaves."""

import copy
import pickle
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from r0.modules.priors.lib.activation_sampling import finite_number
from r0.modules.priors.lib.causal_limits import CausalLimits
from r0.modules.priors.lib.frozen_function_primitives import FrozenTreeFunction
from r0.modules.priors.lib.joint_function_admission import admit_tree_function
from r0.modules.priors.lib.joint_function_calibration import (
    FunctionCalibrationSample,
    freeze_function_calibration,
)
from r0.modules.priors.lib.joint_function_plans import JointFunctionPlan
from r0.modules.priors.lib.joint_point_realization import frozen_matrix
from r0.modules.priors.lib.joint_sampling_records import JointSamplingRecord
from r0.modules.priors.lib.weight_geometry import Matrix, matrix_shape

type IndexTable = tuple[tuple[int, ...], ...]


def tree_feature_weights(sample: Matrix) -> tuple[float, ...]:
    """NumPy float64 population deviation with the source floor/nonfinite fallback."""
    with np.errstate(over="ignore", invalid="ignore"):
        std = np.asarray(sample, dtype=np.float64).std(axis=0, ddof=0)
    weights = np.where(np.isfinite(std), np.maximum(std, 1e-8), 1e-8)
    return tuple(float(value) for value in weights)


def tree_equation(
    plan: JointFunctionPlan,
    calibration: FunctionCalibrationSample,
    features: IndexTable,
    rows: IndexTable,
    leaves: tuple[Matrix, ...],
) -> FrozenTreeFunction:
    thresholds = tuple(
        tuple(
            calibration.sample[row][feature]
            for row, feature in zip(selected_rows, selected_features, strict=True)
        )
        for selected_rows, selected_features in zip(rows, features, strict=True)
    )
    return FrozenTreeFunction(plan.input_width, features, thresholds, leaves)


@dataclass(frozen=True, slots=True)
class JointTreeFunctionRealization:
    plan: JointFunctionPlan
    calibration: FunctionCalibrationSample
    feature_weights: tuple[float, ...]
    features: IndexTable
    threshold_rows: IndexTable
    leaf_gaussian: tuple[Matrix, ...]
    function: FrozenTreeFunction
    version: str = "joint-tree-function-realization-v1"

    def __post_init__(self) -> None:
        if (
            self.version != "joint-tree-function-realization-v1"
            or type(self.plan) is not JointFunctionPlan
            or self.plan.family != "tree"
            or type(self.calibration) is not FunctionCalibrationSample
            or type(self.function) is not FrozenTreeFunction
        ):
            raise ValueError("tree realization requires its plan, calibration and frozen equation")
        plan = self.plan
        if matrix_shape(self.calibration.sample) != (plan.calibration_rows, plan.input_width):
            raise ValueError("tree calibration differs from its planned geometry")
        if (
            type(self.feature_weights) is not tuple
            or len(self.feature_weights) != plan.input_width
            or any(not finite_number(weight) for weight in self.feature_weights)
            or self.feature_weights != tree_feature_weights(self.calibration.sample)
        ):
            raise ValueError("tree feature masses differ from the retained calibration sample")
        values = dict(plan.parameters)
        trees, depth = int(values["trees"]), int(values["depth"])
        for table, high in (
            (self.features, plan.input_width),
            (self.threshold_rows, plan.calibration_rows),
        ):
            if (
                type(table) is not tuple
                or len(table) != trees
                or any(type(row) is not tuple or len(row) != depth for row in table)
                or any(
                    type(index) is not int or not 0 <= index < high
                    for row in table
                    for index in row
                )
            ):
                raise ValueError("tree selections require in-range integer indices at every split")
        if (
            type(self.leaf_gaussian) is not tuple
            or len(self.leaf_gaussian) != trees
            or any(
                matrix_shape(table) != (2**depth, plan.output_width) for table in self.leaf_gaussian
            )
        ):
            raise ValueError("tree leaf draws differ from their planned geometry")
        expected = tree_equation(
            plan, self.calibration, self.features, self.threshold_rows, self.leaf_gaussian
        )
        if self.function != expected:
            raise ValueError("frozen tree equation differs from its retained fit construction")

    def validate(self, decisions: JointSamplingRecord) -> None:
        self.plan.validate(decisions)


def materialize_joint_tree(
    plan: JointFunctionPlan,
    decisions: JointSamplingRecord,
    *,
    rng: np.random.Generator,
    fit_rng: np.random.Generator | None,
    calibration: NDArray[np.float64] | None,
    calibration_id: str,
    limits: CausalLimits,
) -> JointTreeFunctionRealization:
    if type(fit_rng) is not np.random.Generator or fit_rng.bit_generator is rng.bit_generator:
        raise ValueError("tree fitting requires a separate injected fit RNG")
    admit_tree_function(plan, decisions, limits)
    sample = freeze_function_calibration(plan, calibration, calibration_id)
    raw_before = pickle.dumps(rng.bit_generator.state)
    fit_before = pickle.dumps(fit_rng.bit_generator.state)
    raw, fit = copy.deepcopy(rng), copy.deepcopy(fit_rng)
    weights = tree_feature_weights(sample.sample)
    probabilities = np.asarray(weights, dtype=np.float64)
    probabilities = probabilities / probabilities.max()
    probabilities = probabilities / probabilities.sum()
    parameters = dict(plan.parameters)
    trees, depth = int(parameters["trees"]), int(parameters["depth"])
    features = tuple(
        tuple(int(v) for v in row)
        for row in fit.choice(plan.input_width, size=(trees, depth), replace=True, p=probabilities)
    )
    rows = tuple(
        tuple(int(v) for v in row)
        for row in fit.integers(plan.calibration_rows, size=(trees, depth))
    )
    leaves = tuple(
        frozen_matrix(table) for table in raw.normal(size=(trees, 2**depth, plan.output_width))
    )
    equation = tree_equation(plan, sample, features, rows, leaves)
    result = JointTreeFunctionRealization(plan, sample, weights, features, rows, leaves, equation)
    result.validate(decisions)
    if (
        pickle.dumps(rng.bit_generator.state) != raw_before
        or pickle.dumps(fit_rng.bit_generator.state) != fit_before
    ):
        raise ValueError("caller RNG changed during tree construction")
    rng.bit_generator.state = raw.bit_generator.state
    fit_rng.bit_generator.state = fit.bit_generator.state
    return result
