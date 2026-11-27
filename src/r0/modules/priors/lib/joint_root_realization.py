"""Root bases followed by frozen functions, fitted on a separately sampled stream."""

import copy
import pickle
from dataclasses import dataclass

import numpy as np
import torch

from r0.modules.priors.lib.causal_limits import CausalLimits
from r0.modules.priors.lib.frozen_function_evaluation import evaluate_admitted_function
from r0.modules.priors.lib.joint_discretization_realization import (
    JointDiscretizationFunctionRealization,
)
from r0.modules.priors.lib.joint_em_realization import JointEMFunctionRealization
from r0.modules.priors.lib.joint_function_calibration import (
    FunctionCalibrationSample,
    function_calibration_digest,
)
from r0.modules.priors.lib.joint_function_realization import (
    JointMatrixFunctionRealization,
    JointProductFunctionRealization,
    materialize_joint_function,
)
from r0.modules.priors.lib.joint_matrix_realization import bind_weight_batch
from r0.modules.priors.lib.joint_mlp_realization import JointMLPFunctionRealization
from r0.modules.priors.lib.joint_point_realization import frozen_matrix
from r0.modules.priors.lib.joint_rff_realization import JointRFFFunctionRealization
from r0.modules.priors.lib.joint_root_admission import admit_root_realization
from r0.modules.priors.lib.joint_root_base import JointRootBase, JointRootBaseSample, draw_root_base
from r0.modules.priors.lib.joint_root_plans import JointRootPlan
from r0.modules.priors.lib.joint_sampling_records import JointSamplingRecord, validate_name
from r0.modules.priors.lib.joint_tree_realization import JointTreeFunctionRealization
from r0.modules.priors.lib.joint_weight_plans import materialize_joint_weights
from r0.modules.priors.lib.weight_geometry import Matrix, matrix_shape

type RootFunctionRealization = (
    JointMatrixFunctionRealization
    | JointRFFFunctionRealization
    | JointTreeFunctionRealization
    | JointDiscretizationFunctionRealization
    | JointEMFunctionRealization
    | JointMLPFunctionRealization
    | JointProductFunctionRealization
)
_FUNCTION_TYPES = (
    JointMatrixFunctionRealization,
    JointRFFFunctionRealization,
    JointTreeFunctionRealization,
    JointDiscretizationFunctionRealization,
    JointEMFunctionRealization,
    JointMLPFunctionRealization,
    JointProductFunctionRealization,
)
_DEFAULT_LIMITS = CausalLimits()


@dataclass(frozen=True, slots=True)
class JointRootRealization:
    plan: JointRootPlan
    calibration: JointRootBaseSample
    function: RootFunctionRealization
    outputs: Matrix
    calibration_id: str
    version: str = "joint-root-realization-v1"

    def __post_init__(self) -> None:
        validate_name(self.calibration_id, "root calibration identity")
        if (
            self.version != "joint-root-realization-v1"
            or type(self.plan) is not JointRootPlan
            or type(self.calibration) is not JointRootBaseSample
            or type(self.function) not in _FUNCTION_TYPES
            or self.function.plan != self.plan.function
        ):
            raise ValueError(
                "root realization requires its plan, sampled calibration and owned function"
            )
        base, plan = self.calibration.base, self.plan
        if (base.family, base.width, base.covariance_base) != (
            plan.family,
            plan.width,
            plan.covariance_base,
        ):
            raise ValueError("root base differs from its frozen plan")
        if matrix_shape(self.calibration.values) != (plan.calibration_rows, plan.width):
            raise ValueError("root calibration differs from its planned geometry")
        if plan.weights is not None:
            assert base.weights is not None
            bind_weight_batch(plan.weights, (base.weights,))
        if isinstance(
            self.function,
            (
                JointTreeFunctionRealization,
                JointDiscretizationFunctionRealization,
                JointEMFunctionRealization,
                JointMLPFunctionRealization,
                JointProductFunctionRealization,
            ),
        ):
            fitted = self.function.calibration
            if fitted is not None and fitted != FunctionCalibrationSample(
                self.calibration.values,
                self.calibration_id,
                function_calibration_digest(self.calibration.values),
            ):
                raise ValueError("root function fit differs from its retained sampled base inputs")
        # Enclosing sampling/decoding admits the aggregate before these arrays.
        expected = frozen_matrix(
            evaluate_admitted_function(
                self.function.function, torch.tensor(self.calibration.values, dtype=torch.float64)
            ).numpy()
        )
        if (
            matrix_shape(self.outputs) != (plan.calibration_rows, plan.width)
            or self.outputs != expected
        ):
            raise ValueError("root outputs differ from the retained base/function construction")

    def validate(self, decisions: JointSamplingRecord) -> None:
        self.plan.validate(decisions)
        self.function.validate(decisions)


def materialize_joint_root(
    plan: JointRootPlan,
    decisions: JointSamplingRecord,
    *,
    rng: np.random.Generator,
    calibration_rng: np.random.Generator,
    fit_rng: np.random.Generator,
    calibration_id: str,
    limits: CausalLimits = _DEFAULT_LIMITS,
) -> JointRootRealization:
    if (
        type(plan) is not JointRootPlan
        or type(decisions) is not JointSamplingRecord
        or type(limits) is not CausalLimits
    ):
        raise ValueError("root realization requires a frozen plan, decisions and limits")
    generators = (rng, calibration_rng, fit_rng)
    if (
        any(type(generator) is not np.random.Generator for generator in generators)
        or len({id(generator.bit_generator) for generator in generators}) != 3
    ):
        raise ValueError("root construction, calibration and fit require distinct injected RNGs")
    validate_name(calibration_id, "root calibration identity")
    plan.validate(decisions)
    admit_root_realization(plan, decisions, limits)
    before = tuple(pickle.dumps(generator.bit_generator.state) for generator in generators)
    raw, sample_rng, fit = tuple(copy.deepcopy(generator) for generator in generators)
    gaussian, factor, weights = None, None, None
    if plan.family == "covariance":
        gaussian = frozen_matrix(raw.normal(size=(plan.width, plan.width)))
        assert plan.weights is not None
        weights = materialize_joint_weights(plan.weights, decisions, rng=raw, limits=limits)[0]
        with np.errstate(over="ignore", invalid="ignore"):
            factor = frozen_matrix(
                np.asarray(gaussian, dtype=np.float64)
                * np.asarray(weights.values(), dtype=np.float64)[None, :]
            )
    base = JointRootBase(plan.family, plan.width, gaussian, weights, factor, plan.covariance_base)
    calibration = draw_root_base(base, plan.calibration_rows, sample_rng)
    function = materialize_joint_function(
        plan.function,
        decisions,
        rng=raw,
        fit_rng=fit,
        calibration=np.asarray(calibration.values, dtype=np.float64),
        calibration_id=calibration_id,
        limits=limits,
    )
    outputs = frozen_matrix(
        evaluate_admitted_function(
            function.function, torch.tensor(calibration.values, dtype=torch.float64)
        ).numpy()
    )
    result = JointRootRealization(plan, calibration, function, outputs, calibration_id)
    result.validate(decisions)
    if tuple(pickle.dumps(generator.bit_generator.state) for generator in generators) != before:
        raise ValueError("caller RNG changed during root construction")
    for original, staged in zip(generators, (raw, sample_rng, fit), strict=True):
        original.bit_generator.state = staged.bit_generator.state
    return result
