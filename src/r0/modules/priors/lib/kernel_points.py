"""Frozen covariance and Gaussian-mixture point clouds for kernel projections."""

import math
from dataclasses import dataclass
from decimal import Decimal, localcontext
from typing import Any

from r0.modules.priors.lib.activation_sampling import finite_number
from r0.modules.priors.lib.weight_geometry import Matrix, matrix_shape
from r0.modules.priors.lib.weight_vectors import (
    PolynomialWeightVector,
    UniformWeightVector,
    WeightVector,
    weight_vector_from_dict,
)


@dataclass(frozen=True, slots=True)
class KernelPointCloud:
    centers: Matrix = ((0.0,),)
    factors: tuple[Matrix, ...] = (((1.0,),),)
    factor_weights: tuple[WeightVector, ...] = (UniformWeightVector(),)
    component_weights: WeightVector = UniformWeightVector()
    assignments: tuple[int, ...] = (0,)
    innovations: Matrix = ((0.0,),)
    scale_mean: float = 0.0
    scale_coefficient: float = 0.0
    scale_draws: tuple[float, ...] = (0.0,)
    law: str = "gaussian-mixture"
    base_law: str = "gaussian"
    version: str = "kernel-point-cloud-v1"
    exact_terms_limit: int = 1_000_000

    def __post_init__(self) -> None:
        components, dimension = matrix_shape(self.centers)
        points, width = matrix_shape(self.innovations)
        if self.version != "kernel-point-cloud-v1" or width != dimension:
            raise ValueError("kernel point cloud dimensions or version disagree")
        if (
            type(self.factors) is not tuple
            or len(self.factors) != components
            or any(matrix_shape(f) != (dimension, dimension) for f in self.factors)
        ):
            raise ValueError("kernel point factors must be square and match every component")
        allowed = (UniformWeightVector, PolynomialWeightVector)
        if (
            type(self.factor_weights) is not tuple
            or len(self.factor_weights) != components
            or any(type(w) not in allowed or len(w.draws) != dimension for w in self.factor_weights)
            or type(self.component_weights) not in allowed
            or len(self.component_weights.draws) != components
        ):
            raise ValueError("kernel point weights must match components and factor columns")
        if (
            type(self.assignments) is not tuple
            or len(self.assignments) != points
            or any(type(i) is not int or not 0 <= i < components for i in self.assignments)
        ):
            raise ValueError("kernel point assignments must select an existing component")
        if (
            type(self.scale_draws) is not tuple
            or len(self.scale_draws) != components
            or any(
                not finite_number(v)
                for v in (self.scale_mean, self.scale_coefficient, *self.scale_draws)
            )
        ):
            raise ValueError("kernel point scale parameters must be finite and match components")
        if self.law not in ("gaussian-mixture", "random-covariance") or self.base_law not in (
            "gaussian",
            "uniform",
        ):
            raise ValueError("unsupported kernel point law")
        if self.base_law == "uniform" and any(
            not -1 <= v < 1 for row in self.innovations for v in row
        ):
            raise ValueError("uniform kernel point innovations must be in [-1,1)")
        if self.law == "gaussian-mixture" and self.base_law != "gaussian":
            raise ValueError("Gaussian-mixture kernel points require Gaussian innovations")
        if self.law == "random-covariance" and (
            components != 1
            or any(v != 0 for row in self.centers for v in row)
            or any(v != 0 for v in (self.scale_mean, self.scale_coefficient, *self.scale_draws))
        ):
            raise ValueError("covariance points require one centered component without log scaling")
        if type(self.exact_terms_limit) is not int or not 1 <= self.exact_terms_limit < 2**63:
            raise ValueError("exact point cloud terms require a positive limit below 2**63")

    @property
    def shape(self) -> tuple[int, int]:
        return len(self.innovations), len(self.innovations[0])

    @property
    def parameter_count(self) -> int:
        return (
            sum(len(row) for row in (*self.centers, *self.innovations))
            + sum(len(row) for factor in self.factors for row in factor)
            + sum(w.parameter_count for w in self.factor_weights)
            + self.component_weights.parameter_count
            + len(self.assignments)
            + 2
            + len(self.scale_draws)
        )

    def component_probabilities(self) -> tuple[float, ...]:
        weights = self.component_weights.values()
        largest = max(weights)
        if largest == 0:
            raise ValueError(
                "kernel mixture component weights have zero mass; draw was not retried"
            )
        scaled = tuple(v / largest for v in weights)
        total = math.fsum(scaled)
        return tuple(v / total for v in scaled)

    def materialize(self) -> Matrix:
        points, dimension = self.shape
        if points * dimension * dimension > self.exact_terms_limit:
            raise ValueError("kernel point exact term limit exceeded")
        probabilities = self.component_probabilities()
        if any(probabilities[i] == 0 for i in self.assignments):
            raise ValueError("kernel point assignment selects a zero-mass component")
        weights = tuple(w.values() for w in self.factor_weights)
        rows: list[tuple[float, ...]] = []
        with localcontext() as context:
            context.prec = 3 * (309 + 1074) + len(str(dimension)) + 2
            context.Emin, context.Emax = -999999, 999999
            logs = tuple(
                Decimal.from_float(float(self.scale_mean))
                + Decimal.from_float(float(self.scale_coefficient)) * Decimal.from_float(float(z))
                for z in self.scale_draws
            )
            for component, innovation in zip(self.assignments, self.innovations, strict=True):
                rows.append(
                    tuple(
                        _point_value(
                            sum(
                                (
                                    Decimal.from_float(float(a))
                                    * Decimal.from_float(float(w))
                                    * Decimal.from_float(float(z))
                                    for a, w, z in zip(
                                        factor_row, weights[component], innovation, strict=True
                                    )
                                ),
                                Decimal(0),
                            ),
                            center,
                            logs[component],
                        )
                        for factor_row, center in zip(
                            self.factors[component], self.centers[component], strict=True
                        )
                    )
                )
        return tuple(rows)


def _point_value(core: Decimal, center: float, log_scale: Decimal) -> float:
    if not core or log_scale < -10000:
        return float(center)
    # A finite binary64 triple product/sum cannot offset an exponent this large.
    if log_scale > 10000:
        raise ValueError("nonfinite kernel point; construction was not retried")
    with localcontext() as quick:
        quick.prec = 80
        multiplier = log_scale.exp()
    shifted = core * multiplier + Decimal.from_float(float(center))
    if center and abs(shifted) < abs(Decimal.from_float(float(center))) * Decimal("1e-60"):
        # Preserve tiny residuals when a nearly unit scale cancels the center.
        shifted = core * log_scale.exp() + Decimal.from_float(float(center))
    result = float(shifted)
    if not math.isfinite(result):
        raise ValueError("nonfinite kernel point; construction was not retried")
    return result


def point_cloud_from_dict(data: dict[str, Any]) -> KernelPointCloud:
    fields = dict(data)
    for name in ("centers", "innovations"):
        if name in fields:
            fields[name] = tuple(tuple(row) for row in fields[name])
    if "factors" in fields:
        fields["factors"] = tuple(tuple(tuple(row) for row in f) for f in fields["factors"])
    for name in ("assignments", "scale_draws"):
        if name in fields:
            fields[name] = tuple(fields[name])
    if "factor_weights" in fields:
        fields["factor_weights"] = tuple(
            weight_vector_from_dict(w) for w in fields["factor_weights"]
        )
    if "component_weights" in fields:
        fields["component_weights"] = weight_vector_from_dict(fields["component_weights"])
    return KernelPointCloud(**fields)
