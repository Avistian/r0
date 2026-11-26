"""Recorded nonnegative weight-vector laws, separate from realized matrix entries."""

import math
from dataclasses import dataclass
from decimal import Decimal, localcontext
from typing import Any

from r0.modules.priors.lib.activation_sampling import finite_number


@dataclass(frozen=True, slots=True)
class UniformWeightVector:
    draws: tuple[float, ...] = (0.5,)
    version: str = "uniform-weight-vector-v1"

    def __post_init__(self) -> None:
        if self.version != "uniform-weight-vector-v1" or (
            type(self.draws) is not tuple
            or not self.draws
            or any(not finite_number(v) or not 0 <= v < 1 for v in self.draws)
        ):
            raise ValueError("uniform weight vectors require a nonempty finite tuple in [0,1)")

    @property
    def parameter_count(self) -> int:
        return len(self.draws)

    def values(self) -> tuple[float, ...]:
        return self.draws

    def log_values(self) -> tuple[float, ...]:
        return tuple(math.log(v) if v else -math.inf for v in self.draws)

    def centered_logs(self, active: tuple[int, ...]) -> tuple[float, ...]:
        _validate_active(active, len(self.draws))
        selected = set(active)
        logs = self.log_values()
        maximum = max((logs[i] for i in active), default=-math.inf)
        return tuple(
            v - maximum if i in selected and maximum != -math.inf else -math.inf
            for i, v in enumerate(logs)
        )


@dataclass(frozen=True, slots=True)
class PolynomialWeightVector:
    """Permuted polynomial decay with recorded log-Gaussian perturbations, RMS one."""

    draws: tuple[float, ...] = (0.0,)
    decay: float = 1.0
    log_noise_scale: float = 1.0
    permutation: tuple[int, ...] = ()
    version: str = "polynomial-weight-vector-v1"
    exact_scores_limit: int = 1_000_000

    def __post_init__(self) -> None:
        if self.version != "polynomial-weight-vector-v1" or (
            type(self.draws) is not tuple
            or not self.draws
            or any(not finite_number(v) for v in self.draws)
        ):
            raise ValueError("polynomial weight vectors require nonempty finite draws")
        if any(not finite_number(v) or v < 0 for v in (self.decay, self.log_noise_scale)):
            raise ValueError("polynomial decay and log-noise scale must be finite nonnegative")
        if type(self.permutation) is not tuple or (
            self.permutation
            and (
                any(type(i) is not int for i in self.permutation)
                or sorted(self.permutation) != list(range(len(self.draws)))
            )
        ):
            raise ValueError("weight permutation must contain each draw index exactly once")
        if type(self.exact_scores_limit) is not int or not 1 <= self.exact_scores_limit < 2**63:
            raise ValueError("exact weight scores require a positive limit below 2**63")

    @property
    def parameter_count(self) -> int:
        return len(self.draws) + 2 + len(self.permutation)

    def centered_logs(self, active: tuple[int, ...]) -> tuple[float, ...]:
        """Center before rounding, so a huge common perturbation cannot erase decay."""
        n = len(self.draws)
        _validate_active(active, n)
        selected = set(active)
        if n > self.exact_scores_limit:
            raise ValueError("polynomial exact score limit exceeded")
        with localcontext() as context:
            # Two binary64 products, subtraction and centering fit in this bound.
            context.prec = 2 * (309 + 1074) + len(str(n)) + 2
            context.Emin, context.Emax = -999999, 999999
            decay = Decimal.from_float(float(self.decay))
            scale = Decimal.from_float(float(self.log_noise_scale))
            order = self.permutation or tuple(range(n))
            scores = tuple(
                scale * Decimal.from_float(float(self.draws[i]))
                - decay * Decimal.from_float(math.log(i + 1))
                for i in order
            )
            maximum = max((scores[i] for i in active), default=Decimal(0))
            return tuple(
                float(v - maximum) if i in selected else -math.inf for i, v in enumerate(scores)
            )

    def log_values(self) -> tuple[float, ...]:
        logs = self.centered_logs(tuple(range(len(self.draws))))
        offset = 0.5 * (math.log(len(logs)) - math.log(math.fsum(math.exp(2 * v) for v in logs)))
        return tuple(v + offset for v in logs)

    def values(self) -> tuple[float, ...]:
        return tuple(math.exp(v) for v in self.log_values())


def _validate_active(active: tuple[int, ...], width: int) -> None:
    if type(active) is not tuple or any(type(i) is not int or not 0 <= i < width for i in active):
        raise ValueError("active weight coordinates must be an in-range tuple of integer indices")


type WeightVector = UniformWeightVector | PolynomialWeightVector


def weight_vector_from_dict(data: dict[str, Any]) -> WeightVector:
    fields = dict(data)
    if "draws" in fields:
        fields["draws"] = tuple(fields["draws"])
    if fields.get("version") == "polynomial-weight-vector-v1":
        if "permutation" in fields:
            fields["permutation"] = tuple(fields["permutation"])
        return PolynomialWeightVector(**fields)
    return UniformWeightVector(**fields)
