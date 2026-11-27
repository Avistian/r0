"""Atomic source-order realizations of frozen shared point plans."""

import copy
import math

import numpy as np
from numpy.typing import NDArray

from r0.modules.priors.lib.causal_limits import CausalLimits
from r0.modules.priors.lib.joint_point_plans import JointPointPlan
from r0.modules.priors.lib.joint_sampling_records import JointSamplingRecord
from r0.modules.priors.lib.joint_weight_plans import materialize_joint_weights
from r0.modules.priors.lib.kernel_points import KernelPointCloud
from r0.modules.priors.lib.weight_geometry import Matrix

_DEFAULT_LIMITS = CausalLimits()


def frozen_matrix(values: NDArray[np.float64]) -> Matrix:
    return tuple(tuple(float(value) for value in row) for row in values)


def materialize_joint_points(
    plan: JointPointPlan,
    decisions: JointSamplingRecord,
    *,
    rng: np.random.Generator,
    limits: CausalLimits = _DEFAULT_LIMITS,
) -> KernelPointCloud:
    if type(plan) is not JointPointPlan:
        raise ValueError("point realization requires a frozen joint plan")
    plan.validate(decisions)
    if type(rng) is not np.random.Generator or type(limits) is not CausalLimits:
        raise ValueError("point realization requires injected RNG and frozen limits")
    c, d, n = plan.components, plan.dimension, plan.count
    limits.admit_dimensions(
        (d,),
        ((),),
        rows=n,
        calibration_rows=0,
        features=d,
        outputs=d,
        extra_parameters=plan.realization_parameter_count,
        extra_working_channels=c * (d * d + 3 * d + 3) + 4 * d + 4,
    )
    staged = copy.deepcopy(rng)
    mixture = materialize_joint_weights(plan.mixture_weights, decisions, rng=staged, limits=limits)[
        0
    ]
    weights = mixture.values()
    largest = max(weights)
    if largest == 0:
        raise ValueError("point mixture has zero mass; construction was not retried")
    scaled = tuple(v / largest for v in weights)
    total = math.fsum(scaled)
    probabilities = tuple(v / total for v in scaled)
    assignments = tuple(int(i) for i in staged.choice(c, size=n, p=probabilities))
    centers = frozen_matrix(staged.normal(size=(c, d)))
    factors = tuple(frozen_matrix(factor) for factor in staged.normal(size=(c, d, d)))
    factor_weights = materialize_joint_weights(
        plan.factor_weights, decisions, rng=staged, limits=limits
    )
    mean = float(staged.normal())
    coefficient = float(staged.normal())
    scales = tuple(float(v) for v in staged.normal(size=c))
    innovations = frozen_matrix(staged.normal(size=(n, d)))
    cloud = KernelPointCloud(
        centers=centers,
        factors=factors,
        factor_weights=factor_weights,
        component_weights=mixture,
        assignments=assignments,
        innovations=innovations,
        scale_mean=mean,
        scale_coefficient=coefficient,
        scale_draws=scales,
    )
    # Detect arithmetic/assignment failures before committing any caller randomness.
    cloud.materialize()
    rng.bit_generator.state = staged.bit_generator.state
    return cloud
