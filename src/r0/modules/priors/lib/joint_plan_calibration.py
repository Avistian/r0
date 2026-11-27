"""Fit frozen converter plans on caller-owned independent one-channel samples."""

import copy

import numpy as np
from numpy.typing import NDArray

from r0.modules.priors.lib.calibrated_activations import calibrate_source_activation
from r0.modules.priors.lib.calibrated_transforms import calibrate_kumaraswamy
from r0.modules.priors.lib.causal_activations import CausalActivation
from r0.modules.priors.lib.causal_limits import CausalLimits
from r0.modules.priors.lib.joint_activation_plans import JointActivationPlan
from r0.modules.priors.lib.joint_converter_plans import JointConverterPlan
from r0.modules.priors.lib.joint_sampling_records import JointSamplingRecord
from r0.modules.priors.lib.node_pipeline import NumericalConverter

_DEFAULT_LIMITS = CausalLimits()


def fit_joint_activation(
    plan: JointActivationPlan,
    decisions: JointSamplingRecord,
    calibration: NDArray[np.float64] | None,
    *,
    rng: np.random.Generator,
    calibration_id: str,
    limits: CausalLimits = _DEFAULT_LIMITS,
) -> CausalActivation:
    """Fit a standard wrapper atomically; a plain plan needs no calibration sample."""
    if type(plan) is not JointActivationPlan:
        raise ValueError("activation fitting requires a frozen joint plan")
    plan.validate(decisions)
    if type(rng) is not np.random.Generator or type(limits) is not CausalLimits:
        raise ValueError("activation fitting requires injected RNG and frozen limits")
    if plan.selector == "plain":
        return plan.activation
    if calibration is None:
        raise ValueError("standard activation fitting requires a calibration sample")
    assert plan.scale is not None
    staged_rng = copy.deepcopy(rng)
    result = calibrate_source_activation(
        plan.activation,
        calibration,
        rng=staged_rng,
        scale=plan.scale,
        calibration_id=calibration_id,
        limits=limits,
    )
    rng.bit_generator.state = staged_rng.bit_generator.state
    return result


def fit_joint_converter(
    plan: JointConverterPlan,
    decisions: JointSamplingRecord,
    calibration: NDArray[np.float64],
    *,
    rng: np.random.Generator,
    calibration_id: str,
    limits: CausalLimits = _DEFAULT_LIMITS,
) -> NumericalConverter:
    """Validate bindings, admit fit geometry, and commit shift RNG only on success.

    The caller supplies the selected channel at the actual ancestral pipeline
    stage. This helper cannot certify that provenance or population moments.
    Limits cover this fit and its output converter; graph-wide retained plans,
    decisions and live ancestors require the graph fitter's aggregate admission.
    """
    if type(plan) is not JointConverterPlan:
        raise ValueError("converter fitting requires a frozen joint plan")
    plan.validate(decisions)
    if type(rng) is not np.random.Generator or type(limits) is not CausalLimits:
        raise ValueError("converter fitting requires injected RNG and frozen limits")
    if (
        type(calibration) is not np.ndarray
        or calibration.ndim != 2
        or calibration.shape[0] < 1
        or calibration.shape[1] != 1
        or calibration.dtype.kind != "f"
    ):
        raise ValueError("converter fitting requires a nonempty floating one-channel sample")
    if type(calibration_id) is not str or not calibration_id:
        raise ValueError("converter calibration identity must be nonempty")
    parameters, working = 1, 0
    if plan.shapes is not None:
        parameters, working = 8, 16
    elif plan.activation is not None:
        parameters, working = 9 + plan.activation.activation.parameter_count, 10
    limits.admit_dimensions(
        (1,),
        ((),),
        rows=calibration.shape[0],
        calibration_rows=0,
        features=1,
        outputs=1,
        extra_parameters=parameters,
        extra_working_channels=working,
    )
    if not np.isfinite(calibration).all():
        raise ValueError("converter calibration sample must be finite")
    staged_rng = copy.deepcopy(rng)
    transform = CausalActivation()
    if plan.shapes is not None:
        transform = calibrate_kumaraswamy(
            calibration,
            a=plan.shapes[0],
            b=plan.shapes[1],
            calibration_id=calibration_id,
            limits=limits,
        )
    elif plan.activation is not None:
        transform = fit_joint_activation(
            plan.activation,
            decisions,
            calibration,
            rng=staged_rng,
            calibration_id=calibration_id,
            limits=limits,
        )
    converter = NumericalConverter(plan.channel, transform, plan.prior.variant)
    rng.bit_generator.state = staged_rng.bit_generator.state
    return converter
