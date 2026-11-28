"""Independent fixed converter fits with center, embedding and post-function custody."""

from dataclasses import dataclass

import numpy as np
import torch
from numpy.typing import NDArray

from r0.modules.priors.lib.calibrated_activations import calibrate_standardization
from r0.modules.priors.lib.categorical_converter import CategoricalConverter
from r0.modules.priors.lib.categorical_converter_plans import CategoricalConverterPlan
from r0.modules.priors.lib.causal_limits import CausalLimits
from r0.modules.priors.lib.joint_function_admission import function_record_cost
from r0.modules.priors.lib.joint_function_calibration import (
    FunctionCalibrationSample,
    function_calibration_digest,
)
from r0.modules.priors.lib.joint_function_realization import materialize_joint_function
from r0.modules.priors.lib.joint_point_realization import frozen_matrix
from r0.modules.priors.lib.joint_root_realization import (
    JointRootRealization,
    RootFunctionRealization,
    materialize_joint_root,
)
from r0.modules.priors.lib.joint_sampling_records import JointSamplingRecord
from r0.modules.priors.lib.joint_weight_plans import materialize_joint_weights
from r0.modules.priors.lib.raw_draw_capture import RawDrawLedger, recorded_draw
from r0.modules.priors.lib.weight_vectors import WeightVector


@dataclass(frozen=True, slots=True)
class CategoricalConverterRealization:
    plan: CategoricalConverterPlan
    calibration: FunctionCalibrationSample
    converter: CategoricalConverter
    permutation: tuple[int, ...] = ()
    embeddings: JointRootRealization | None = None
    weights: WeightVector | None = None
    function: RootFunctionRealization | None = None
    raw_start: int = 0
    raw_stop: int = 0
    version: str = "independent-category-converter-realization-v1"


def fit_categorical_converter(
    plan: CategoricalConverterPlan,
    decisions: JointSamplingRecord,
    *,
    name: str,
    start: int,
    values: NDArray[np.float64],
    identity: str,
    streams: dict[str, np.random.Generator],
    capture: RawDrawLedger,
    limits: CausalLimits,
) -> CategoricalConverterRealization:
    plan.validate(decisions)
    if (
        type(values) is not np.ndarray
        or values.shape != (plan.calibration_rows, plan.width)
        or values.dtype.kind != "f"
    ):
        raise ValueError("converter fit requires its declared independent population matrix")
    function_record_cost((plan, values.shape), limits)
    work = 128 * plan.calibration_rows * len(plan.vocabulary) * (plan.width + 8)
    if work > limits.working_bytes:
        raise ValueError("converter fit distance/probability workspace exceeds declared limit")
    sample = frozen_matrix(values)
    calibration = FunctionCalibrationSample(sample, identity, function_calibration_digest(sample))
    raw_start = len(capture.freeze())
    permutation: tuple[int, ...] = ()
    embeddings, weights, function = None, None, None
    mean, scale, bias = (), (), ()
    if plan.embeddings is not None:
        embeddings = materialize_joint_root(
            plan.embeddings,
            decisions,
            rng=streams["raw"],
            calibration_rng=streams["calibration"],
            fit_rng=streams["fit"],
            calibration_id=identity + "-embeddings",
            limits=limits,
            capture=capture,
        )
        representatives = embeddings.outputs
        assert plan.weights is not None
        weights = materialize_joint_weights(
            plan.weights, decisions, rng=streams["raw"], limits=limits, capture=capture
        )[0]
        standard = calibrate_standardization(
            values, calibration_id=identity + "-standardization", limits=limits
        )
        mean = standard.mean
        scale = tuple(1.0 / v for v in standard.inverse_std)
        bias = tuple(float(v) for v in np.log(np.asarray(weights.values()) + 1e-4))
    else:
        permutation = tuple(
            int(i)
            for i in recorded_draw(
                streams["fit"],
                capture,
                owner=plan.operation,
                role="center-permutation",
                distribution="permutation",
                shape=(plan.calibration_rows,),
                parameters=(("count", plan.calibration_rows),),
            )
        )
        representatives = tuple(sample[i] for i in permutation[: len(plan.vocabulary)])
        if plan.function is not None:
            assignment = CategoricalConverter(
                name,
                start,
                start + plan.width,
                plan.vocabulary,
                "neighbor_disc",
                representatives,
                exponent=plan.exponent,
            ).apply(torch.tensor(values, dtype=torch.float64))
            # Fitting population includes repeated selected centers with original row multiplicity.
            function = materialize_joint_function(
                plan.function,
                decisions,
                rng=streams["raw"],
                fit_rng=streams["fit"],
                calibration=assignment.state.numpy(),
                calibration_id=identity + "-post-function",
                limits=limits,
                capture=capture,
            )
    converter = CategoricalConverter(
        name,
        start,
        start + plan.width,
        plan.vocabulary,
        plan.mode,
        representatives,
        plan.exponent,
        mean,
        scale,
        bias,
        plan.factor,
        function.function if function is not None else None,
    )
    return CategoricalConverterRealization(
        plan,
        calibration,
        converter,
        permutation,
        embeddings,
        weights,
        function,
        raw_start,
        len(capture.freeze()),
    )
