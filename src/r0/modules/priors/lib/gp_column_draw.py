"""Finite joint RBF feature columns; no jitter, fitting, retries or row independence."""

from dataclasses import dataclass

import numpy as np

from r0.modules.priors.lib.activation_sampling import finite_number
from r0.modules.priors.lib.causal_limits import CausalLimits
from r0.modules.priors.lib.weight_geometry import Matrix, matrix_shape

_DEFAULT_LIMITS = CausalLimits()


@dataclass(frozen=True, slots=True)
class GPColumnKernel:
    length_scale: float = 0.1
    amplitude: float = 1.0
    version: str = "gp-column-rbf-v1"

    def __post_init__(self) -> None:
        if self.version != "gp-column-rbf-v1" or any(
            not finite_number(v) or v <= 0 for v in (self.length_scale, self.amplitude)
        ):
            raise ValueError("GP column kernel needs finite positive scale and amplitude")


def admit_gp_geometry(
    rows: int,
    columns: int,
    dimension: int,
    kernels: int,
    limits: CausalLimits,
    *,
    extra_numbers: int = 0,
) -> dict[str, int]:
    if (
        type(limits) is not CausalLimits
        or any(type(v) is not int for v in (rows, columns, dimension, kernels, extra_numbers))
        or not (1 <= rows <= 1024 and 1 <= columns <= 256 and 1 <= dimension <= 32)
    ):
        raise ValueError("GP geometry requires 1..1024 rows, 1..256 columns, 1..32 dimensions")
    if kernels not in (1, columns) or extra_numbers < 0:
        raise ValueError("GP kernels must be shared or one per column")
    numbers = 4 * kernels * rows * rows + 8 * rows * (columns + dimension) + extra_numbers
    working = 512 * numbers + 8 * 1024**2
    if columns > min(limits.node_channels, limits.total_channels) or numbers > limits.parameters:
        raise ValueError("GP population parameter/channel limit exceeded")
    if working > limits.working_bytes:
        raise ValueError("GP population working memory limit exceeded")
    return {"parameters": numbers, "estimated_working_bytes": working}


@dataclass(frozen=True, slots=True)
class GPColumnDraw:
    locations: Matrix
    innovations: Matrix
    kernels: tuple[GPColumnKernel, ...]
    covariance: tuple[Matrix, ...]
    factors: tuple[Matrix, ...]
    values: Matrix
    version: str = "gp-column-draw-v1"


def build_gp_columns(
    locations: Matrix,
    innovations: Matrix,
    *,
    kernels: tuple[GPColumnKernel, ...] = (GPColumnKernel(),),
    limits: CausalLimits = _DEFAULT_LIMITS,
) -> GPColumnDraw:
    n, d = matrix_shape(locations)
    m, c = matrix_shape(innovations)
    if m != n or type(kernels) is not tuple or any(type(k) is not GPColumnKernel for k in kernels):
        raise ValueError("GP population rows/kernels disagree")
    admit_gp_geometry(n, c, d, len(kernels), limits)
    x, z = np.asarray(locations), np.asarray(innovations)
    covariance: list[Matrix] = []
    factors: list[Matrix] = []
    result = np.empty((n, c), dtype=np.float64)
    for j, kernel in enumerate(kernels):
        # One coordinate at a time avoids an undeclared n*n*dimension temporary.
        distance = np.zeros((n, n), dtype=np.float64)
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            for axis in range(d):
                delta = (x[:, axis, None] - x[None, :, axis]) / kernel.length_scale
                distance += delta * delta
            k = np.float64(kernel.amplitude) * kernel.amplitude * np.exp(-0.5 * distance)
        if not np.isfinite(k).all() or not np.isfinite(distance).all():
            raise ValueError("nonfinite GP covariance geometry; no clipping or retry")
        try:
            factor = np.linalg.cholesky(k)
        except np.linalg.LinAlgError as error:
            raise ValueError(
                "GP covariance is not positive definite; no jitter or retry"
            ) from error
        tolerance = 128 * np.finfo(float).eps * n * max(1.0, float(k.max()))
        if float(np.max(np.abs(factor @ factor.T - k))) > tolerance:
            raise ValueError("GP Cholesky reconstruction exceeds roundoff bound")
        if len(kernels) == 1:
            result[:] = factor @ z
        else:
            result[:, j] = factor @ z[:, j]
        covariance.append(gp_matrix(k))
        factors.append(gp_matrix(factor))
    if not np.isfinite(result).all():
        raise ValueError("nonfinite GP population; no retry")
    return GPColumnDraw(
        locations, innovations, kernels, tuple(covariance), tuple(factors), gp_matrix(result)
    )


def gp_matrix(values: np.ndarray) -> Matrix:
    return tuple(tuple(float(v) for v in row) for row in values)
