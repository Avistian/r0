"""Fresh retained rows from a frozen root; emission never recalibrates its function."""

import copy
import pickle
from dataclasses import dataclass

import numpy as np
import torch

from r0.modules.priors.lib.causal_limits import CausalLimits
from r0.modules.priors.lib.frozen_function_evaluation import evaluate_admitted_function
from r0.modules.priors.lib.joint_function_admission import function_record_cost
from r0.modules.priors.lib.joint_point_realization import frozen_matrix
from r0.modules.priors.lib.joint_root_admission import (
    admit_root_realization,
    function_evaluation_channels,
)
from r0.modules.priors.lib.joint_root_base import JointRootBaseSample, draw_root_base
from r0.modules.priors.lib.joint_root_realization import JointRootRealization
from r0.modules.priors.lib.joint_sampling_records import JointSamplingRecord, validate_name
from r0.modules.priors.lib.weight_geometry import Matrix, matrix_shape

_DEFAULT_LIMITS = CausalLimits()


@dataclass(frozen=True, slots=True)
class JointRootEmission:
    root: JointRootRealization
    sample: JointRootBaseSample
    outputs: Matrix
    sample_id: str
    version: str = "joint-root-emission-v1"

    def __post_init__(self) -> None:
        validate_name(self.sample_id, "root emission identity")
        if (
            self.version != "joint-root-emission-v1"
            or type(self.root) is not JointRootRealization
            or type(self.sample) is not JointRootBaseSample
            or self.sample.base != self.root.calibration.base
        ):
            raise ValueError("root emission requires its frozen root and matching sampled base")
        if self.sample_id == self.root.calibration_id:
            raise ValueError("root emission identity must differ from its calibration identity")
        # The caller admits the entire retained root plus fresh buffers before construction.
        expected = frozen_matrix(
            evaluate_admitted_function(
                self.root.function.function, torch.tensor(self.sample.values, dtype=torch.float64)
            ).numpy()
        )
        if (
            matrix_shape(self.outputs) != matrix_shape(self.sample.values)
            or self.outputs != expected
        ):
            raise ValueError("root emission outputs differ from its fixed function")


def admit_root_emission(
    root: JointRootRealization,
    decisions: JointSamplingRecord,
    rows: int,
    limits: CausalLimits,
) -> None:
    """Include retained root and duplicated serialized base, plus fresh evaluation buffers."""
    if type(rows) is not int or not 1 <= rows < 2**63:
        raise ValueError("root emission requires positive bounded integer rows")
    root_report = admit_root_realization(root.plan, decisions, limits)
    metadata, storage = function_record_cost((root, root.calibration.base, decisions), limits)
    width = root.plan.width
    parameters = metadata + 16 + rows * (3 * width + int(root.plan.family == "ball"))
    report = limits.admit_dimensions(
        (width,),
        ((),),
        rows=rows,
        calibration_rows=0,
        features=width,
        outputs=width,
        extra_parameters=parameters,
        extra_working_channels=24 * width + function_evaluation_channels(root.plan.function),
    )
    working = (
        report["estimated_working_bytes"]
        + root_report["estimated_working_bytes"]
        + storage
        - 64 * metadata
        + 1024
    )
    if working > limits.working_bytes:
        raise ValueError("aggregate root emission working memory limit exceeded")


def sample_joint_root_values(
    root: JointRootRealization,
    decisions: JointSamplingRecord,
    *,
    rows: int,
    rng: np.random.Generator,
    sample_id: str,
    limits: CausalLimits = _DEFAULT_LIMITS,
) -> JointRootEmission:
    """Commit injected RNG only after a finite, admitted emission is fully validated."""
    if (
        type(root) is not JointRootRealization
        or type(decisions) is not JointSamplingRecord
        or type(limits) is not CausalLimits
        or type(rng) is not np.random.Generator
    ):
        raise ValueError("root emission requires a frozen root, decisions, limits and injected RNG")
    validate_name(sample_id, "root emission identity")
    if sample_id == root.calibration_id:
        raise ValueError("root emission identity must differ from its calibration identity")
    root.validate(decisions)
    admit_root_emission(root, decisions, rows, limits)
    before = pickle.dumps(rng.bit_generator.state)
    staged = copy.deepcopy(rng)
    sample = draw_root_base(root.calibration.base, rows, staged)
    outputs = frozen_matrix(
        evaluate_admitted_function(
            root.function.function, torch.tensor(sample.values, dtype=torch.float64)
        ).numpy()
    )
    result = JointRootEmission(root, sample, outputs, sample_id)
    if pickle.dumps(rng.bit_generator.state) != before:
        raise ValueError("caller RNG changed during root emission")
    rng.bit_generator.state = staged.bit_generator.state
    return result
