"""Frozen empirical rank and quantile maps fitted on an explicit numeric sample."""

import hashlib
import math
from dataclasses import dataclass
from typing import Any, Literal, cast

import numpy as np
import torch
from numpy.typing import NDArray

Kind = Literal["rank", "quantile-uniform", "quantile-normal"]


def _sample(values: NDArray[np.float64]) -> NDArray[np.float64]:
    if (
        type(values) is not np.ndarray
        or values.dtype != np.float64
        or values.ndim != 1
        or not 1 <= len(values) <= 20000
        or not np.isfinite(values).all()
    ):
        raise ValueError("quantile maps require 1..20000 finite float64 values")
    return values


@dataclass(frozen=True, slots=True)
class FittedQuantileMap:
    kind: Kind
    knots: tuple[float, ...]
    counts: tuple[int, ...]
    tail_probability: float
    calibration_sha256: str
    version: str = "r0-fitted-quantile-map-v1"

    def __post_init__(self) -> None:
        if (
            self.version != "r0-fitted-quantile-map-v1"
            or self.kind not in ("rank", "quantile-uniform", "quantile-normal")
            or type(self.knots) is not tuple
            or type(self.counts) is not tuple
            or not 1 <= len(self.knots) == len(self.counts) <= 20000
            or any(type(k) not in (float, int) or not math.isfinite(k) for k in self.knots)
            or any(a >= b for a, b in zip(self.knots, self.knots[1:], strict=False))
            or any(type(c) is not int or c <= 0 for c in self.counts)
            or sum(self.counts) > 20000
            or type(self.tail_probability) not in (float, int)
            or not math.isfinite(self.tail_probability)
            or not 0 <= self.tail_probability < 0.5
            or (self.kind == "quantile-normal" and self.tail_probability < np.finfo(np.float64).eps)
            or (self.kind != "quantile-normal" and self.tail_probability != 0)
            or type(self.calibration_sha256) is not str
            or len(self.calibration_sha256) != 64
            or any(c not in "0123456789abcdef" for c in self.calibration_sha256)
        ):
            raise ValueError("invalid frozen empirical quantile map")

    def apply(self, values: NDArray[np.float64]) -> NDArray[np.float64]:
        values = _sample(values)
        knots = np.asarray(self.knots, dtype=np.float64)
        counts = np.asarray(self.counts, dtype=np.int64)
        cumulative = np.concatenate((np.array([0], dtype=np.int64), np.cumsum(counts)))
        total = int(cumulative[-1])
        if self.kind == "rank":
            left = np.searchsorted(knots, values, side="left")
            right = np.searchsorted(knots, values, side="right")
            return (cumulative[left] + cumulative[right]).astype(np.float64) / (2 * total)
        midpoints = (cumulative[1:] - counts / 2) / total
        probabilities = _interpolate(values, knots, midpoints)
        if self.kind == "quantile-uniform":
            return probabilities
        clipped = np.clip(probabilities, self.tail_probability, 1 - self.tail_probability)
        return (
            math.sqrt(2) * torch.erfinv(torch.tensor(2 * clipped - 1, dtype=torch.float64))
        ).numpy()


def _interpolate(
    values: NDArray[np.float64], knots: NDArray[np.float64], midpoints: NDArray[np.float64]
) -> NDArray[np.float64]:
    if len(knots) == 1:
        return np.where(values < knots[0], 0.0, np.where(values > knots[0], 1.0, midpoints[0]))
    index = np.clip(np.searchsorted(knots, values, side="right") - 1, 0, len(knots) - 2)
    low, high = knots[index], knots[index + 1]
    bounded = np.clip(values, knots[0], knots[-1])
    with np.errstate(over="ignore"):
        width = high - low
    scale = np.where(np.isinf(width), 0.5, 1.0)
    fraction = (bounded * scale - low * scale) / (high * scale - low * scale)
    result = midpoints[index] + fraction * (midpoints[index + 1] - midpoints[index])
    return np.where(values < knots[0], 0.0, np.where(values > knots[-1], 1.0, result))


def fit_quantile_map(
    calibration: NDArray[np.float64], *, kind: Kind, tail_probability: float | None = None
) -> FittedQuantileMap:
    """Fit only the supplied sample. Callers own context/calibration provenance.

    No labels, query rows, random draws or hidden refit are accessed. Normal maps
    default to tail clipping at 0.5 / (calibration count + 1).
    """
    calibration = _sample(calibration)
    knots, counts = np.unique(calibration, return_counts=True)
    tail = 0.5 / (len(calibration) + 1) if kind == "quantile-normal" else 0.0
    if tail_probability is not None:
        tail = tail_probability
    return FittedQuantileMap(
        kind,
        tuple(float(k) for k in knots),
        tuple(int(c) for c in counts),
        tail,
        hashlib.sha256(calibration.astype("<f8", copy=False).tobytes()).hexdigest(),
    )


def fitted_quantile_map_from_dict(value: Any) -> FittedQuantileMap:
    if type(value) is not dict:
        raise ValueError("quantile map must be an object")
    data = cast(dict[str, Any], value).copy()
    if set(data) != {
        "kind",
        "knots",
        "counts",
        "tail_probability",
        "calibration_sha256",
        "version",
    }:
        raise ValueError("quantile map fields differ")
    for name in ("knots", "counts"):
        if type(data[name]) is not list:
            raise ValueError("quantile map knots and counts must be arrays")
        data[name] = tuple(cast(list[Any], data[name]))
    return FittedQuantileMap(**data)
