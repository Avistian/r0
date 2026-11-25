"""Stationary covariance kernels and explicit finite kernel combinations."""

import math
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True, slots=True)
class StationaryKernel:
    kind: str = "rbf"
    length_scale: float = 1.0
    axes: tuple[int, ...] = ()
    feature_scales: tuple[float, ...] = ()
    weight: float = 1.0

    def __post_init__(self) -> None:
        if self.kind not in ("rbf", "matern12", "matern32", "matern52"):
            raise ValueError("unsupported stationary kernel")
        for value in (self.length_scale, self.weight, *self.feature_scales):
            if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
                raise ValueError("kernel scales and weights must be finite and positive")
        if type(self.axes) is not tuple or any(type(v) is not int or v < 0 for v in self.axes):
            raise ValueError("kernel axes must be a tuple of nonnegative integers")
        if len(set(self.axes)) != len(self.axes) or type(self.feature_scales) is not tuple:
            raise ValueError("kernel axes must be unique and feature_scales must be a tuple")


def kernel_covariance(
    support: NDArray[np.float64],
    kernels: tuple[StationaryKernel, ...],
    combine: str,
) -> tuple[NDArray[np.float64], list[dict[str, object]]]:
    weights = np.asarray([k.weight for k in kernels], dtype=np.float64)
    if combine == "sum":
        weights /= weights.max()
        weights /= weights.sum()
        if np.any(weights <= 0):
            raise ValueError("kernel weight normalization underflow; task was not retried")
    covariance = (
        np.zeros((len(support), len(support)))
        if combine == "sum"
        else np.ones((len(support), len(support)))
    )
    cards: list[dict[str, object]] = []
    for kernel, weight in zip(kernels, weights, strict=True):
        axes = kernel.axes or tuple(range(support.shape[1]))
        multipliers = kernel.feature_scales or (1.0,) * len(axes)
        with np.errstate(over="ignore", under="ignore", invalid="ignore"):
            lengths = np.asarray(multipliers) * kernel.length_scale
        if not np.isfinite(lengths).all() or np.any(lengths <= 0):
            raise ValueError("nonfinite or zero kernel length scales; task was not retried")
        distance = np.zeros_like(covariance)
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            for axis, length in zip(axes, lengths, strict=True):
                difference = support[:, axis, None] - support[None, :, axis]
                normalized = difference / length
                overflow = np.isinf(difference)
                if overflow.any():
                    # Opposite-sign coordinates can overflow before a large divisor.
                    scaled = support[:, axis] / length
                    fallback = scaled[:, None] - scaled[None, :]
                    normalized[overflow] = fallback[overflow]
                distance = np.hypot(distance, normalized)
            if kernel.kind == "rbf":
                values = np.exp(-0.5 * distance**2)
            else:
                nu2 = {"matern12": 1, "matern32": 3, "matern52": 5}[kernel.kind]
                t = math.sqrt(nu2) * distance
                log_polynomial = np.zeros_like(t)
                if nu2 == 3:
                    log_polynomial = np.log1p(t)
                elif nu2 == 5:
                    log_t = np.log(t)
                    log_polynomial = np.logaddexp(0, np.logaddexp(log_t, 2 * log_t - math.log(3)))
                values = np.exp(log_polynomial - t)
                values[np.isinf(t)] = 0  # The limiting covariance is zero.
        if not np.isfinite(values).all():
            raise ValueError("nonfinite kernel covariance; task was not retried")
        if combine == "sum":
            covariance += weight * values
        else:
            covariance *= values
        cards.append(
            {
                "kind": kernel.kind,
                "axes": list(axes),
                "length_scales": lengths.tolist(),
                "weight": float(weight),
            }
        )
    return covariance, cards
