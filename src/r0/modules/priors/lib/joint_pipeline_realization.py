"""Fit sampled pipelines at actual ancestral outputs using the construction ledger."""

from dataclasses import dataclass

import numpy as np

from r0.modules.priors.lib.calibrated_activations import (
    calibrate_standardization,
    reconstruct_source_activation,
)
from r0.modules.priors.lib.calibrated_transforms import (
    calibrate_kumaraswamy,
    calibrate_l2_normalization,
)
from r0.modules.priors.lib.causal_activations import CausalActivation
from r0.modules.priors.lib.causal_limits import CausalLimits
from r0.modules.priors.lib.functional_causal_evaluation import sample_functional_prefix
from r0.modules.priors.lib.functional_causal_node import FunctionalCausalNode
from r0.modules.priors.lib.functional_prefix import FunctionalPrefix
from r0.modules.priors.lib.joint_function_calibration import (
    FunctionCalibrationSample,
    function_calibration_digest,
)
from r0.modules.priors.lib.joint_pipeline_plans import JointPipelinePlan
from r0.modules.priors.lib.joint_point_realization import frozen_matrix
from r0.modules.priors.lib.joint_sampling_records import JointSamplingRecord
from r0.modules.priors.lib.joint_weight_plans import materialize_joint_weights
from r0.modules.priors.lib.node_pipeline import CausalNodePipeline, NumericalConverter
from r0.modules.priors.lib.raw_draw_capture import RawDrawLedger, recorded_draw
from r0.modules.priors.lib.weight_vectors import WeightVector


@dataclass(frozen=True, slots=True)
class JointPipelineRealization:
    plan: JointPipelinePlan
    calibration: FunctionCalibrationSample
    weights: WeightVector | None
    pipeline: CausalNodePipeline
    raw_start: int
    raw_stop: int
    version: str = "independent-node-pipeline-realization-v1"


def fit_node_pipeline(
    plan: JointPipelinePlan,
    decisions: JointSamplingRecord,
    nodes: tuple[FunctionalCausalNode, ...],
    *,
    rows: int,
    streams: dict[str, np.random.Generator],
    capture: RawDrawLedger,
    limits: CausalLimits,
    retained: tuple[object, ...],
) -> JointPipelineRealization:
    """The final prefix node is its frozen function/root, without later stages.

    Ancestors already contain their final pipelines and innovations. These fresh
    rows are independent of mechanism fitting, emitted rows and target adaptation.
    """
    plan.validate(decisions)
    identity = f"pipeline-node-{len(nodes) - 1}-fit"
    prefix = FunctionalPrefix(nodes, rows, limits, retained)
    start = len(capture.freeze())
    population = sample_functional_prefix(
        prefix,
        rng=streams["calibration"],
        owner=(*plan.operation, "calibration"),
        capture=capture,
    )
    value = population.states[-1]
    sample = frozen_matrix(value.numpy())
    calibration = FunctionCalibrationSample(sample, identity, function_calibration_digest(sample))
    stages = ()
    importance: tuple[float, ...] = ()
    weights = None
    if plan.weights is not None:
        weights = materialize_joint_weights(
            plan.weights,
            decisions,
            rng=streams["raw"],
            limits=limits,
            capture=capture,
        )[0]
        factors = np.asarray(weights.values(), dtype=np.float64)
        maximum = factors.max()
        if not np.isfinite(factors).all() or maximum <= 0:
            raise ValueError("pipeline channel weights have zero or nonfinite RMS")
        scaled = factors / maximum
        importance = tuple(float(v) for v in scaled / np.sqrt(np.mean(scaled**2)))
        standard = calibrate_standardization(
            value.numpy(),
            calibration_id=identity + "-standardize",
            limits=limits,
        )
        stages = (standard,)
        value = standard.apply(value) * value.new_tensor(importance)
    l2 = None
    if plan.prior.l2_normalization:
        l2 = calibrate_l2_normalization(
            value.numpy(), calibration_id=identity + "-l2", limits=limits
        )
        value = l2.apply(value)
    converters: list[NumericalConverter] = []
    for converter in plan.converters:
        channel = value[:, converter.channel : converter.channel + 1].numpy()
        transform: CausalActivation = CausalActivation()
        converter_id = identity + f"-converter-{converter.channel}"
        if converter.shapes is not None:
            transform = calibrate_kumaraswamy(
                channel,
                a=converter.shapes[0],
                b=converter.shapes[1],
                calibration_id=converter_id,
                limits=limits,
            )
        elif converter.activation is not None:
            activation = converter.activation
            shift = recorded_draw(
                streams["fit"],
                capture,
                owner=activation.operation,
                role="activation-shift-rows",
                distribution="integers",
                shape=(1,),
                parameters=(("high", rows),),
            )
            assert activation.scale is not None
            transform = reconstruct_source_activation(
                activation.activation,
                channel,
                shift_rows=(int(shift[0]),),
                scale=activation.scale,
                calibration_id=converter_id,
            )
        converters.append(NumericalConverter(converter.channel, transform, converter.prior.variant))
    pipeline = CausalNodePipeline(
        plan.width,
        stages,
        importance,
        l2,
        tuple(converters),
        plan.importance,
    )
    return JointPipelineRealization(
        plan,
        calibration,
        weights,
        pipeline,
        start,
        len(capture.freeze()),
    )
