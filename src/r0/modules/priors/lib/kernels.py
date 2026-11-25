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
        _validate_geometry(self.length_scale, self.axes, self.feature_scales, self.weight)


@dataclass(frozen=True, slots=True)
class RationalQuadraticKernel:
    kind: str = "rational_quadratic"
    length_scale: float = 1.0
    axes: tuple[int, ...] = ()
    feature_scales: tuple[float, ...] = ()
    weight: float = 1.0
    alpha: float = 1.0

    def __post_init__(self) -> None:
        if self.kind != "rational_quadratic":
            raise ValueError("unsupported rational quadratic kernel")
        _validate_geometry(self.length_scale, self.axes, self.feature_scales, self.weight)
        if type(self.alpha) not in (int, float) or not math.isfinite(self.alpha) or self.alpha <= 0:
            raise ValueError("alpha must be finite and positive")


@dataclass(frozen=True, slots=True)
class SpectralKernel:
    kind: str = "spectral"
    length_scale: float = 1.0
    axes: tuple[int, ...] = ()
    feature_scales: tuple[float, ...] = ()
    weight: float = 1.0
    frequencies: tuple[float, ...] = ()

    def __post_init__(self) -> None:
        if self.kind != "spectral":
            raise ValueError("unsupported spectral kernel")
        _validate_geometry(self.length_scale, self.axes, self.feature_scales, self.weight)
        if type(self.frequencies) is not tuple or any(
            type(v) not in (int, float) or not math.isfinite(v) or v < 0 for v in self.frequencies
        ):
            raise ValueError(
                "frequencies must be a tuple of finite nonnegative cycles per input unit"
            )


type Kernel = StationaryKernel | RationalQuadraticKernel | SpectralKernel


def _validate_geometry(
    length_scale: float, axes: tuple[int, ...], feature_scales: tuple[float, ...], weight: float
) -> None:
    for value in (length_scale, weight, *feature_scales):
        if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
            raise ValueError("kernel scales and weights must be finite and positive")
    if type(axes) is not tuple or any(type(v) is not int or v < 0 for v in axes):
        raise ValueError("kernel axes must be a tuple of nonnegative integers")
    if len(set(axes)) != len(axes) or type(feature_scales) is not tuple:
        raise ValueError("kernel axes must be unique and feature_scales must be a tuple")


def kernel_covariance(
    support: NDArray[np.float64],
    kernels: tuple[Kernel, ...],
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
            if isinstance(kernel, RationalQuadraticKernel):
                if not np.isfinite(distance).all():
                    raise ValueError(
                        "unrepresentable rational quadratic distance; task was not retried"
                    )
                # log(alpha * log1p(r^2 / (2*alpha))) avoids squaring r,
                # overflowing 2*alpha, and losing the RBF limit for large alpha.
                log_ratio = 2 * np.log(distance) - math.log(2) - math.log(kernel.alpha)
                log_softplus = np.log(np.logaddexp(0, log_ratio))
                log_softplus = np.where(log_ratio < -36, log_ratio, log_softplus)
                values = np.exp(-np.exp(math.log(kernel.alpha) + log_softplus))
            elif kernel.kind == "rbf" or isinstance(kernel, SpectralKernel):
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
        if isinstance(kernel, SpectralKernel):
            frequencies = kernel.frequencies or (0.0,) * len(axes)
            for axis, frequency in zip(axes, frequencies, strict=True):
                if frequency == 0:
                    continue
                with np.errstate(over="ignore", invalid="ignore"):
                    delta = support[:, axis, None] - support[None, :, axis]
                    cycles = delta * frequency
                    overflow = np.isinf(delta)
                    if overflow.any():
                        scaled = support[:, axis] * frequency
                        fallback = scaled[:, None] - scaled[None, :]
                        cycles[overflow] = fallback[overflow]
                # Declared cap: float64 spacing at 2**40 cycles is 1/4096 of a cycle.
                if not np.isfinite(cycles).all() or np.any(np.abs(cycles) > 2**40):
                    raise ValueError(
                        "spectral phase exceeds the declared numerical limit; task was not retried"
                    )
                values *= np.cos(2 * math.pi * np.remainder(cycles, 1))
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
        if isinstance(kernel, RationalQuadraticKernel):
            cards[-1]["alpha"] = kernel.alpha
        if isinstance(kernel, SpectralKernel):
            cards[-1]["frequencies"] = list(kernel.frequencies or (0.0,) * len(axes))
            cards[-1]["frequency_units"] = "cycles per input unit"
    return covariance, cards
