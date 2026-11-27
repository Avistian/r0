"""Ordered MLP fits with retained stage inputs, constants and construction ancestry."""

import copy
import hashlib
import json
import pickle
from dataclasses import dataclass

import numpy as np
import torch
from numpy.typing import NDArray

from r0.modules.priors.lib.calibrated_activations import (
    CalibratedActivation,
    calibrate_source_activation,
    reconstruct_source_activation,
)
from r0.modules.priors.lib.causal_limits import CausalLimits
from r0.modules.priors.lib.frozen_function_composites import FrozenMLPFunction
from r0.modules.priors.lib.frozen_function_primitives import FrozenLinearFunction
from r0.modules.priors.lib.joint_activation_plans import JointActivationPlan
from r0.modules.priors.lib.joint_function_admission import admit_mlp_function
from r0.modules.priors.lib.joint_function_calibration import (
    FunctionCalibrationSample,
    freeze_function_calibration,
    function_calibration_digest,
)
from r0.modules.priors.lib.joint_function_plans import JointFunctionPlan
from r0.modules.priors.lib.joint_matrix_plans import JointMatrixPlan
from r0.modules.priors.lib.joint_matrix_realization import (
    JointMatrixRealization,
    materialize_joint_matrix,
)
from r0.modules.priors.lib.joint_point_realization import frozen_matrix
from r0.modules.priors.lib.joint_sampling_records import JointSamplingRecord
from r0.modules.priors.lib.weight_geometry import Matrix, matrix_shape


def mlp_stage_sample(
    plan: JointFunctionPlan,
    calibration: FunctionCalibrationSample,
    index: int,
    value: torch.Tensor,
) -> FunctionCalibrationSample:
    key = json.dumps([calibration.identity, plan.operation, index], separators=(",", ":")).encode()
    identity = "mlp-stage-" + hashlib.sha256(key).hexdigest()
    sample = frozen_matrix(value.numpy())
    return FunctionCalibrationSample(sample, identity, function_calibration_digest(sample))


@dataclass(frozen=True, slots=True)
class JointMLPFunctionRealization:
    plan: JointFunctionPlan
    calibration: FunctionCalibrationSample
    matrices: tuple[JointMatrixRealization, ...]
    activation_inputs: tuple[FunctionCalibrationSample, ...]
    outputs: Matrix
    function: FrozenMLPFunction
    version: str = "joint-mlp-function-realization-v1"

    def __post_init__(self) -> None:
        if (
            self.version != "joint-mlp-function-realization-v1"
            or type(self.plan) is not JointFunctionPlan
            or self.plan.family != "mlp"
            or type(self.calibration) is not FunctionCalibrationSample
            or type(self.function) is not FrozenMLPFunction
            or type(self.matrices) is not tuple
            or len(self.matrices) != len(self.plan.matrices)
            or any(type(matrix) is not JointMatrixRealization for matrix in self.matrices)
            or tuple(matrix.plan for matrix in self.matrices) != self.plan.matrices
            or type(self.activation_inputs) is not tuple
            or len(self.activation_inputs) != len(self.plan.activations)
            or any(
                type(sample) is not FunctionCalibrationSample for sample in self.activation_inputs
            )
        ):
            raise ValueError("MLP realization requires its plan, sample and ordered owned stages")
        plan = self.plan
        if matrix_shape(self.calibration.sample) != (plan.calibration_rows, plan.input_width):
            raise ValueError("MLP calibration differs from its planned geometry")
        parameters = dict(plan.parameters)
        expected = FrozenMLPFunction(
            tuple(FrozenLinearFunction(matrix.materialize()[0]) for matrix in self.matrices),
            self.function.activations,
            bool(parameters["start_activation"]),
            bool(parameters["end_activation"]),
        )
        if self.function != expected:
            raise ValueError("frozen MLP equation differs from its retained construction")
        value = torch.tensor(self.calibration.sample, dtype=torch.float64)
        matrix_index, activation_index = 0, 0
        for child in plan.ordered_children:
            if isinstance(child, JointMatrixPlan):
                value = value @ value.new_tensor(expected.linears[matrix_index].matrix).T
                matrix_index += 1
            else:
                assert isinstance(child, JointActivationPlan)
                retained = self.activation_inputs[activation_index]
                if retained != mlp_stage_sample(plan, self.calibration, activation_index, value):
                    raise ValueError("MLP activation input differs from preceding stage outputs")
                activation = expected.activations[activation_index]
                assert child.scale is not None
                fitted = reconstruct_source_activation(
                    child.activation,
                    value.numpy(),
                    shift_rows=activation.shift_rows,
                    scale=child.scale,
                    calibration_id=retained.identity,
                )
                if activation != fitted:
                    raise ValueError(
                        "MLP activation constants differ from their retained fit construction"
                    )
                value = activation.apply(value)
                activation_index += 1
        if matrix_shape(self.outputs) != (
            plan.calibration_rows,
            plan.output_width,
        ) or self.outputs != frozen_matrix(value.numpy()):
            raise ValueError("MLP outputs differ from the retained stage construction")

    def validate(self, decisions: JointSamplingRecord) -> None:
        self.plan.validate(decisions)
        for matrix in self.matrices:
            matrix.validate(decisions)


def materialize_joint_mlp(
    plan: JointFunctionPlan,
    decisions: JointSamplingRecord,
    *,
    rng: np.random.Generator,
    fit_rng: np.random.Generator | None,
    calibration: NDArray[np.float64] | None,
    calibration_id: str,
    limits: CausalLimits,
) -> JointMLPFunctionRealization:
    if (plan.activations and fit_rng is None) or (
        fit_rng is not None
        and (type(fit_rng) is not np.random.Generator or fit_rng.bit_generator is rng.bit_generator)
    ):
        raise ValueError("MLP fitting requires a separate injected fit RNG")
    admit_mlp_function(plan, decisions, limits)
    sample = freeze_function_calibration(plan, calibration, calibration_id)
    raw_before = pickle.dumps(rng.bit_generator.state)
    fit_before = pickle.dumps(fit_rng.bit_generator.state) if fit_rng is not None else None
    raw, fit = copy.deepcopy(rng), copy.deepcopy(fit_rng)
    value = torch.tensor(sample.sample, dtype=torch.float64)
    matrices: list[JointMatrixRealization] = []
    linears: list[FrozenLinearFunction] = []
    activations: list[CalibratedActivation] = []
    inputs: list[FunctionCalibrationSample] = []
    for child in plan.ordered_children:
        if isinstance(child, JointMatrixPlan):
            matrix = materialize_joint_matrix(child, decisions, rng=raw, limits=limits)
            linear = FrozenLinearFunction(matrix.materialize()[0])
            value = value @ value.new_tensor(linear.matrix).T
            if not torch.isfinite(value).all():
                raise ValueError("nonfinite MLP linear stage output; construction was not retried")
            matrices.append(matrix)
            linears.append(linear)
        else:
            assert isinstance(child, JointActivationPlan)
            assert child.scale is not None
            assert fit is not None
            retained = mlp_stage_sample(plan, sample, len(inputs), value)
            activation = calibrate_source_activation(
                child.activation,
                value.numpy(),
                rng=fit,
                scale=child.scale,
                calibration_id=retained.identity,
                limits=limits,
            )
            value = activation.apply(value)
            inputs.append(retained)
            activations.append(activation)
    parameters = dict(plan.parameters)
    function = FrozenMLPFunction(
        tuple(linears),
        tuple(activations),
        bool(parameters["start_activation"]),
        bool(parameters["end_activation"]),
    )
    result = JointMLPFunctionRealization(
        plan,
        sample,
        tuple(matrices),
        tuple(inputs),
        frozen_matrix(value.numpy()),
        function,
    )
    result.validate(decisions)
    if pickle.dumps(rng.bit_generator.state) != raw_before or (
        fit_rng is not None and pickle.dumps(fit_rng.bit_generator.state) != fit_before
    ):
        raise ValueError("caller RNG changed during MLP construction")
    rng.bit_generator.state = raw.bit_generator.state
    if fit_rng is not None and fit is not None:
        fit_rng.bit_generator.state = fit.bit_generator.state
    return result
