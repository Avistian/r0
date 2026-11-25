"""Estimate signal statistics without squaring extreme raw magnitudes."""

import numpy as np
from numpy.typing import NDArray


def stable_calibration(
    raw: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    if not np.isfinite(raw).all():
        raise ValueError("nonfinite or degenerate calibration; task was not retried")
    magnitude = np.max(np.abs(raw), axis=0)
    if np.any(magnitude == 0):
        raise ValueError("nonfinite or degenerate calibration; task was not retried")
    scaled = raw / magnitude
    center = scaled.mean(axis=0) * magnitude
    scale = scaled.std(axis=0) * magnitude
    if not np.isfinite(center).all() or not np.isfinite(scale).all() or np.any(scale <= 0):
        raise ValueError("nonfinite or degenerate calibration; task was not retried")
    return center, scale
