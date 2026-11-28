"""Frozen root base laws and retained raw draws before the source function."""

import math
from dataclasses import dataclass

import numpy as np

from r0.modules.priors.lib.activation_sampling import finite_number
from r0.modules.priors.lib.frozen_function_primitives import function_width
from r0.modules.priors.lib.joint_point_realization import frozen_matrix
from r0.modules.priors.lib.raw_draw_capture import (
    RawDrawLedger,
    raw_transaction,
    recorded_draw,
)
from r0.modules.priors.lib.weight_geometry import Matrix, matrix_shape
from r0.modules.priors.lib.weight_vectors import (
    PolynomialWeightVector,
    UniformWeightVector,
    WeightVector,
)


@dataclass(frozen=True, slots=True)
class JointRootBase:
    family: str
    width: int
    gaussian_factor: Matrix | None = None
    weights: WeightVector | None = None
    factor: Matrix | None = None
    covariance_base: str | None = None
    version: str = "joint-root-base-v1"

    def __post_init__(self) -> None:
        function_width(self.width)
        if (
            self.version != "joint-root-base-v1"
            or type(self.family) is not str
            or self.family not in ("uniform", "ball", "gaussian", "covariance")
        ):
            raise ValueError("root base requires its version and supported family")
        if self.family != "covariance":
            if any(
                value is not None
                for value in (self.gaussian_factor, self.weights, self.factor, self.covariance_base)
            ):
                raise ValueError("only covariance roots own factor constructions")
            return
        if (
            self.gaussian_factor is None
            or matrix_shape(self.gaussian_factor) != (self.width, self.width)
            or type(self.weights) not in (UniformWeightVector, PolynomialWeightVector)
            or self.weights is None
            or len(self.weights.draws) != self.width
            or self.factor is None
            or matrix_shape(self.factor) != (self.width, self.width)
            or self.covariance_base not in ("uniform", "gaussian")
        ):
            raise ValueError("covariance roots require aligned Gaussian factors, weights and base")
        # Frozen records are validated before enclosing graph admission. Verify
        # each binary64 product directly, without constructing numerical arrays
        # or a duplicate derived matrix before that aggregate gate.
        weights = self.weights.values()
        for row, stored in zip(self.gaussian_factor, self.factor, strict=True):
            for value, weight, actual in zip(row, weights, stored, strict=True):
                expected = float(value) * float(weight)
                if not math.isfinite(expected) or actual != expected:
                    raise ValueError(
                        "root covariance factor differs from its column-weighted construction"
                    )


def root_base_values(
    base: JointRootBase, draws: Matrix, radial_uniform: tuple[float, ...]
) -> Matrix:
    """Transform admitted raw draws, rejecting undefined directions without retries."""
    values = np.asarray(draws, dtype=np.float64)
    family = base.covariance_base if base.family == "covariance" else base.family
    if family == "uniform":
        values = -1 + 2 * values
    elif family == "ball":
        magnitude = np.abs(values).max(axis=1)
        if np.any(magnitude == 0):
            raise ValueError("zero root ball direction; construction was not retried")
        scaled = values / magnitude[:, None]
        unit = scaled / np.sqrt(np.square(scaled).sum(axis=1))[:, None]
        values = unit * np.asarray(radial_uniform, dtype=np.float64)[:, None] ** (1 / base.width)
    if base.family == "covariance":
        assert base.factor is not None
        with np.errstate(over="ignore", invalid="ignore"):
            values = values @ np.asarray(base.factor, dtype=np.float64).T
    result = frozen_matrix(values)
    matrix_shape(result)
    return result


@dataclass(frozen=True, slots=True)
class JointRootBaseSample:
    base: JointRootBase
    draws: Matrix
    radial_uniform: tuple[float, ...]
    values: Matrix
    version: str = "joint-root-base-sample-v1"

    def __post_init__(self) -> None:
        if self.version != "joint-root-base-sample-v1" or type(self.base) is not JointRootBase:
            raise ValueError("root sample requires its version and frozen base")
        rows, width = matrix_shape(self.draws)
        if width != self.base.width or matrix_shape(self.values) != (rows, width):
            raise ValueError("root sample geometry differs from its base")
        uniform = self.base.family == "uniform" or self.base.covariance_base == "uniform"
        if uniform and any(not 0 <= value < 1 for row in self.draws for value in row):
            raise ValueError("root uniform draws must lie in [0,1)")
        if (
            type(self.radial_uniform) is not tuple
            or len(self.radial_uniform) != (rows if self.base.family == "ball" else 0)
            or any(not finite_number(value) or not 0 <= value < 1 for value in self.radial_uniform)
        ):
            raise ValueError("root ball radii require one raw uniform draw per row")
        if self.values != root_base_values(self.base, self.draws, self.radial_uniform):
            raise ValueError("root sample differs from its retained base construction")


def draw_root_base(
    base: JointRootBase,
    rows: int,
    rng: np.random.Generator,
    *,
    capture: RawDrawLedger | None = None,
    owner: tuple[str, ...] = (),
) -> JointRootBaseSample:
    """Draw into a staged RNG after the enclosing root has admitted full geometry."""
    if capture is not None:
        with raw_transaction(rng, capture) as (staged, capture):
            uniform = base.family == "uniform" or base.covariance_base == "uniform"
            draws = frozen_matrix(
                recorded_draw(
                    staged,
                    capture,
                    owner=owner,
                    role="root-base-values",
                    distribution="random" if uniform else "normal",
                    shape=(rows, base.width),
                )
            )
            radii = (
                tuple(
                    map(
                        float,
                        recorded_draw(
                            staged,
                            capture,
                            owner=owner,
                            role="root-base-radii",
                            distribution="random",
                            shape=(rows,),
                        ),
                    )
                )
                if base.family == "ball"
                else ()
            )
            return JointRootBaseSample(base, draws, radii, root_base_values(base, draws, radii))
    uniform = base.family == "uniform" or base.covariance_base == "uniform"
    draws = frozen_matrix(
        rng.random((rows, base.width)) if uniform else rng.normal(size=(rows, base.width))
    )
    radii = tuple(map(float, rng.random(rows))) if base.family == "ball" else ()
    return JointRootBaseSample(base, draws, radii, root_base_values(base, draws, radii))
