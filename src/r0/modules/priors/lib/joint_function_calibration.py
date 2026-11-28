"""Immutable caller-supplied calibration samples with canonical input identities."""

import hashlib
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from r0.modules.priors.lib.joint_function_plans import JointFunctionPlan
from r0.modules.priors.lib.joint_point_realization import frozen_matrix
from r0.modules.priors.lib.joint_sampling_records import validate_name
from r0.modules.priors.lib.weight_geometry import Matrix, matrix_shape


def function_calibration_digest(sample: Matrix) -> str:
    rows, width = matrix_shape(sample)
    canonical = np.asarray(sample, dtype="<f8", order="C")
    digest = hashlib.sha256(f"{rows},{width}:".encode())
    digest.update(memoryview(canonical).cast("B"))
    return digest.hexdigest()


@dataclass(frozen=True, slots=True)
class FunctionCalibrationSample:
    sample: Matrix
    identity: str
    digest: str
    version: str = "joint-function-calibration-v1"

    def __post_init__(self) -> None:
        validate_name(self.identity, "calibration identity")
        if self.version != "joint-function-calibration-v1":
            raise ValueError("unsupported function calibration version")
        if type(self.digest) is not str or self.digest != function_calibration_digest(self.sample):
            raise ValueError("function calibration digest differs from its retained sample")


def freeze_function_calibration(
    plan: JointFunctionPlan,
    sample: NDArray[np.float64] | None,
    identity: str,
) -> FunctionCalibrationSample:
    """Called after family admission. Input independence must be established by the caller."""
    validate_name(identity, "calibration identity")
    if (
        type(sample) is not np.ndarray
        or sample.ndim != 2
        or sample.dtype.kind != "f"
        or sample.shape != (plan.calibration_rows, plan.input_width)
    ):
        raise ValueError("function calibration must match its planned floating sample geometry")
    copied = np.array(sample, dtype=np.float64, order="C", copy=True)
    if not np.isfinite(copied).all():
        raise ValueError("function calibration must be finite")
    frozen = frozen_matrix(copied)
    return FunctionCalibrationSample(frozen, identity, function_calibration_digest(frozen))
