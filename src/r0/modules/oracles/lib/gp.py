"""Small exact Gaussian conditioning with an auditable numerical policy."""

import math
from dataclasses import dataclass
from typing import Literal, cast

import numpy as np
from numpy.typing import NDArray

from r0.modules.episodes import PredictorView

Matrix = tuple[tuple[float, ...], ...]
Array = NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class GPSpec:
    family: Literal["linear", "rbf"]
    mean: float = 0.0
    signal_variance: float = 1.0
    noise_variance: float = 0.0
    length_scale: float | None = None
    jitter: float = 0.0

    def __post_init__(self) -> None:
        if self.family not in ("linear", "rbf"):
            raise ValueError("unsupported GP reference family")
        for value in (self.mean, self.signal_variance, self.noise_variance, self.jitter):
            if type(value) not in (float, int) or not math.isfinite(value):
                raise ValueError("GP parameters must be finite numbers")
        if self.signal_variance <= 0 or self.noise_variance < 0 or self.jitter < 0:
            raise ValueError("signal variance must be positive; noise and jitter nonnegative")
        if self.family == "rbf":
            if (
                type(self.length_scale) not in (float, int)
                or self.length_scale is None
                or not math.isfinite(self.length_scale)
                or self.length_scale <= 0
            ):
                raise ValueError("RBF requires a finite positive length_scale")
        elif self.length_scale is not None:
            raise ValueError("linear kernel does not accept length_scale")


@dataclass(frozen=True, slots=True)
class GPPrediction:
    mean: tuple[float, ...]
    latent_variance: tuple[float, ...]
    observation_variance: tuple[float, ...]
    latent_covariance: Matrix
    observation_covariance: Matrix
    prior_query_covariance: Matrix
    conditioning_covariance: Matrix
    cross_covariance: Matrix
    cholesky: Matrix
    weights: tuple[float, ...]
    raw_latent_covariance: Matrix
    correction_bound: float
    symmetry_error: float
    minimum_eigenvalue: float
    corrections: tuple[str, ...]
    jitter: float
    stabilized: bool


def _matrix(array: Array) -> Matrix:
    return tuple(tuple(float(x) for x in row) for row in array)


def _coordinates(values: tuple[float, ...], *, context: bool = False) -> Array:
    lower = 0 if context else 1
    if type(values) is not tuple or not lower <= len(values) <= 64:
        raise ValueError(f"reference requires {lower}..64 coordinates")
    if any(type(x) not in (float, int) or not math.isfinite(x) for x in values):
        raise ValueError("GP coordinates must be finite numbers")
    return np.asarray(values, dtype=np.float64)


def _kernel(x: Array, y: Array, spec: GPSpec) -> Array:
    if spec.family == "linear":
        return spec.signal_variance * x[:, None] * y[None, :]
    assert spec.length_scale is not None
    distance = (x[:, None] - y[None, :]) / spec.length_scale
    return spec.signal_variance * np.exp(-0.5 * distance * distance)


def _calculate(x: Array, y: Array, queries: Array, spec: GPSpec) -> GPPrediction:
    prior = _kernel(queries, queries, spec)
    conditioning = _kernel(x, x, spec) + (spec.noise_variance + spec.jitter) * np.eye(len(x))
    cross = _kernel(x, queries, spec)
    if len(x):
        factor = np.linalg.cholesky(conditioning)
        weights = np.linalg.solve(factor.T, np.linalg.solve(factor, y - spec.mean))
        mean = spec.mean + cross.T @ weights
        projected = np.linalg.solve(factor, cross)
        raw = prior - projected.T @ projected
    else:
        factor = np.empty((0, 0), dtype=np.float64)
        weights = np.empty(0, dtype=np.float64)
        mean = np.full(len(queries), spec.mean, dtype=np.float64)
        raw = prior.copy()
    bound = 1e-12 * max(1.0, float(np.max(np.diag(prior))))
    symmetry_error = float(np.max(np.abs(raw - raw.T)))
    if symmetry_error > bound:
        raise ValueError("GP covariance asymmetry exceeds numerical correction bound")
    covariance = raw / 2 + raw.T / 2
    corrections: list[str] = []
    if symmetry_error:
        corrections.append("symmetrized latent covariance; original retained")
    for i, value in enumerate(np.diag(covariance)):
        if value < -bound:
            raise ValueError("GP negative variance exceeds numerical correction bound")
        if value < 0:
            corrections.append(f"latent variance[{i}] clamped from {float(value)} to 0")
            covariance[i, i] = 0.0
    minimum_eigenvalue = float(np.linalg.eigvalsh(covariance).min())
    if minimum_eigenvalue < -bound:
        raise ValueError("GP covariance is not positive semidefinite within numerical bound")
    observation = covariance + spec.noise_variance * np.eye(len(queries))
    arrays = (prior, conditioning, cross, factor, weights, mean, raw, covariance, observation)
    if any(not np.isfinite(a).all() for a in arrays):
        raise ValueError("GP calculation exceeds finite numeric range")
    return GPPrediction(
        tuple(float(v) for v in mean),
        tuple(float(v) for v in np.diag(covariance)),
        tuple(float(v) for v in np.diag(observation)),
        _matrix(covariance),
        _matrix(observation),
        _matrix(prior),
        _matrix(conditioning),
        _matrix(cross),
        _matrix(factor),
        tuple(float(v) for v in weights),
        _matrix(raw),
        bound,
        symmetry_error,
        minimum_eigenvalue,
        tuple(corrections),
        spec.jitter,
        spec.jitter > 0,
    )


def _predict(x: Array, y: Array, queries: Array, spec: GPSpec) -> GPPrediction:
    try:
        with np.errstate(over="raise", invalid="raise", divide="raise", under="ignore"):
            return _calculate(x, y, queries, spec)
    except np.linalg.LinAlgError as error:
        raise ValueError(
            "GP conditioning failed: singular/non-positive covariance; "
            "declare positive observation noise or explicit jitter"
        ) from error
    except (FloatingPointError, OverflowError) as error:
        raise ValueError("GP calculation exceeds supported floating-point range") from error


def gp_prior(queries: tuple[float, ...], spec: GPSpec) -> GPPrediction:
    if spec.jitter:
        raise ValueError("prior prediction has no conditioning matrix for jitter")
    return _predict(
        np.empty(0, dtype=np.float64), np.empty(0, dtype=np.float64), _coordinates(queries), spec
    )


def gp(view: PredictorView, spec: GPSpec) -> GPPrediction:
    if (
        view.task_type != "regression"
        or len(view.features) != 1
        or view.features[0].kind != "continuous"
    ):
        raise ValueError("GP requires regression with exactly one continuous feature")
    values = tuple(row.values[0] for row in view.context + view.queries)
    if any(type(x) not in (float, int) for x in values):
        raise ValueError("GP requires fully observed numeric coordinates")
    # Runtime validation above narrows the episode's mixed feature-value domain.
    x = tuple(float(cast(float, v)) for v in values[: len(view.context)])
    queries = tuple(float(cast(float, v)) for v in values[len(view.context) :])
    y = np.asarray(view.context_targets, dtype=np.float64)
    return _predict(_coordinates(x, context=True), y, _coordinates(queries), spec)
