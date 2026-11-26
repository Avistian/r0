"""Finite matrix stages: Gaussian draws, jitter/normalization and flattened activation."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from decimal import Decimal, localcontext

import torch

from r0.modules.priors.lib.activation_sampling import finite_number
from r0.modules.priors.lib.causal_activations import CausalActivation, activate
from r0.modules.priors.lib.weight_matrices import (
    FactorizedWeights,
    ImportanceFactorizedWeights,
    ImportanceWeights,
    LaplaceKernelWeights,
    Matrix,
    matrix_shape,
)


@dataclass(frozen=True, slots=True)
class GaussianWeights:
    draws: Matrix = ((1.0,),)
    version: str = "gaussian-weights-v1"

    def __post_init__(self) -> None:
        matrix_shape(self.draws)
        if self.version != "gaussian-weights-v1":
            raise ValueError("unsupported Gaussian matrix version")

    @property
    def shape(self) -> tuple[int, int]:
        return len(self.draws), len(self.draws[0])

    @property
    def parameter_count(self) -> int:
        return sum(len(row) for row in self.draws)

    def materialize(self) -> Matrix:
        return self.draws


type SimpleWeights = (
    GaussianWeights
    | FactorizedWeights
    | ImportanceFactorizedWeights
    | ImportanceWeights
    | LaplaceKernelWeights
)
SIMPLE_WEIGHT_TYPES = (
    GaussianWeights,
    FactorizedWeights,
    ImportanceFactorizedWeights,
    ImportanceWeights,
    LaplaceKernelWeights,
)


@dataclass(frozen=True, slots=True)
class NormalizedWeights:
    source: SimpleWeights | TransformedWeights = GaussianWeights()
    jitter: Matrix = ((0.0,),)
    jitter_scale: float = 1e-6
    epsilon: float = 1e-6
    version: str = "normalized-weights-v1"
    exact_terms_limit: int = 1_000_000

    def __post_init__(self) -> None:
        if self.version != "normalized-weights-v1" or type(self.source) not in (
            *SIMPLE_WEIGHT_TYPES,
            TransformedWeights,
        ):
            raise ValueError("normalization requires a raw finite matrix construction")
        _validate_stage(self.shape, self.jitter, self.jitter_scale, self.exact_terms_limit)
        if not finite_number(self.epsilon) or self.epsilon < 0:
            raise ValueError("normalization epsilon must be finite nonnegative")

    @property
    def shape(self) -> tuple[int, int]:
        return self.source.shape

    @property
    def parameter_count(self) -> int:
        return self.source.parameter_count + sum(len(row) for row in self.jitter) + 2

    def materialize(self) -> Matrix:
        _admit_terms(self.shape, self.exact_terms_limit)
        return _finish(self.source.materialize(), self.jitter, self.jitter_scale, self.epsilon)


@dataclass(frozen=True, slots=True)
class TransformedWeights:
    child: NormalizedWeights = field(default_factory=NormalizedWeights)
    activation: CausalActivation = CausalActivation()
    jitter: Matrix = ((0.0,),)
    jitter_scale: float = 1e-3
    version: str = "transformed-weights-v1"
    exact_terms_limit: int = 1_000_000

    def __post_init__(self) -> None:
        if (
            self.version != "transformed-weights-v1"
            or type(self.child) is not NormalizedWeights
            or type(self.child.source) not in SIMPLE_WEIGHT_TYPES
            or type(self.activation) is not CausalActivation
        ):
            raise ValueError("activation matrices require a normalized nonrecursive simple child")
        _validate_stage(self.shape, self.jitter, self.jitter_scale, self.exact_terms_limit)

    @property
    def shape(self) -> tuple[int, int]:
        return self.child.shape

    @property
    def parameter_count(self) -> int:
        return (
            self.child.parameter_count
            + len(self.activation.parameters)
            + sum(len(row) for row in self.jitter)
            + 1
        )

    def materialize(self) -> Matrix:
        _admit_terms(self.shape, self.exact_terms_limit)
        # One synthetic row couples all matrix entries for rank/softmax/etc.
        child = torch.tensor(self.child.materialize(), dtype=torch.float64).reshape(1, -1)
        transformed = activate(child, self.activation).reshape(self.shape)
        values = tuple(tuple(float(v) for v in row) for row in transformed)
        return _finish(values, self.jitter, self.jitter_scale, None)


def _validate_stage(shape: tuple[int, int], jitter: Matrix, scale: float, limit: int) -> None:
    if matrix_shape(jitter) != shape:
        raise ValueError("matrix jitter dimensions must match the source")
    if not finite_number(scale) or scale < 0:
        raise ValueError("matrix jitter scale must be finite nonnegative")
    if type(limit) is not int or not 1 <= limit < 2**63:
        raise ValueError("exact matrix stage terms require a positive limit below 2**63")


def _admit_terms(shape: tuple[int, int], limit: int) -> None:
    if shape[0] * shape[1] > limit:
        raise ValueError("exact matrix stage term limit exceeded")


def _finish(base: Matrix, jitter: Matrix, scale: float, epsilon: float | None) -> Matrix:
    rows: list[tuple[float, ...]] = []
    with localcontext() as context:
        context.prec = 2 * (309 + 1074) + 2
        context.Emin, context.Emax = -999999, 999999
        coefficient = Decimal.from_float(float(scale))
        for values, noise in zip(base, jitter, strict=True):
            exact = tuple(
                Decimal.from_float(float(v)) + coefficient * Decimal.from_float(float(z))
                for v, z in zip(values, noise, strict=True)
            )
            if epsilon is None:
                result = tuple(float(v) for v in exact)
            else:
                floor = Decimal.from_float(float(epsilon))
                largest = max(floor, *(abs(v) for v in exact))
                if largest == 0:
                    result = (0.0,) * len(exact)
                else:
                    scaled = tuple(float(v / largest) for v in exact)
                    denominator = float(floor / largest) + math.hypot(*scaled)
                    result = tuple(v / denominator for v in scaled)
            if any(not math.isfinite(v) for v in result):
                raise ValueError("nonfinite matrix stage output; construction was not retried")
            rows.append(result)
    return tuple(rows)
