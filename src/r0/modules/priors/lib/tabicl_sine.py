"""Random-frequency sine with explicit independent frozen input calibration."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

from r0.modules.priors.lib.causal_limits import CausalLimits

if TYPE_CHECKING:
    from r0.modules.priors.lib.calibrated_activations import CalibratedStandardization
    from r0.modules.priors.lib.causal_activations import CausalActivation


_DEFAULT = CausalLimits()


def sample_tabicl_sine(
    calibration: NDArray[np.float64],
    *,
    calibration_id: str,
    rng: np.random.Generator,
    min_frequency: float = 0.1,
    max_frequency: float = 100.0,
    limits: CausalLimits = _DEFAULT,
) -> tuple[CalibratedStandardization, CausalActivation]:
    """Return standardization then sine; bounds customize the source defaults."""
    from r0.modules.priors.lib.calibrated_activations import calibrate_standardization
    from r0.modules.priors.lib.causal_activations import CausalActivation
    from r0.modules.priors.lib.raw_draw_capture import raw_transactions

    if (
        any(
            type(v) not in (int, float) or not math.isfinite(v) or v <= 0
            for v in (min_frequency, max_frequency)
        )
        or min_frequency > max_frequency
    ):
        raise ValueError("sine frequency bounds must be finite, positive and ordered")
    standard = calibrate_standardization(
        calibration, calibration_id=calibration_id, profile="tabicl", limits=limits
    )
    if standard.parameter_count + 2 > limits.parameters:
        raise ValueError("sine calibration and parameters exceed construction limit")
    with raw_transactions({"parameters": rng}, None) as (streams, _):
        frequency = math.exp(
            float(streams["parameters"].uniform(math.log(min_frequency), math.log(max_frequency)))
        )
        phase = float(streams["parameters"].uniform(0, 2 * math.pi))
        return standard, CausalActivation(
            "sin", (("input_scale", frequency), ("input_shift", phase))
        )
