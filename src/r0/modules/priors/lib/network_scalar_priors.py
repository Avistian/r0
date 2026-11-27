"""Named scalar hyperlaws: independent paper TNLU versus pinned relative-sigma code."""

import math
from dataclasses import dataclass
from statistics import NormalDist

import numpy as np

from r0.modules.priors.lib.activation_sampling import finite_number
from r0.modules.priors.lib.raw_draw_capture import RawDrawLedger

_TNLU = ("p1-paper-tnlu-v1", "tabicl-9530bd6-tnlu-v1")


@dataclass(frozen=True, slots=True)
class NetworkScalarPrior:
    family: str = "fixed"
    low: float = 1.0
    high: float = 1.0
    offset: float = 0.0
    rounded: bool = False
    choices: tuple[float, ...] = ()
    version: str = "network-scalar-prior-v1"

    def __post_init__(self) -> None:
        if (
            self.version != "network-scalar-prior-v1"
            or self.family
            not in (*_TNLU, "fixed", "choice", "integer-uniform", "uniform", "log-uniform")
            or any(not finite_number(v) for v in (self.low, self.high, self.offset))
            or self.low > self.high
            or type(self.rounded) is not bool
            or type(self.choices) is not tuple
            or len(self.choices) > 256
            or any(not finite_number(v) for v in self.choices)
        ):
            raise ValueError("invalid network scalar prior")
        if self.family == "choice":
            if not self.choices or (self.low, self.high) != (1.0, 1.0):
                raise ValueError("choice prior requires values and unused default bounds")
        elif self.choices:
            raise ValueError("only choice priors accept choices")
        if self.family == "fixed" and self.low != self.high:
            raise ValueError("fixed prior requires identical bounds")
        if self.family not in ("fixed", "choice", "integer-uniform") and self.low >= self.high:
            raise ValueError("continuous scalar laws require increasing bounds")
        if self.family in (*_TNLU, "log-uniform") and self.low <= 0:
            raise ValueError("log-uniform hyperparameters require positive bounds")
        if self.family == "integer-uniform" and any(
            v != int(v) or abs(v) >= 2**53 for v in (self.low, self.high)
        ):
            raise ValueError("integer law bounds must be exactly representable integers")


@dataclass(frozen=True, slots=True)
class NetworkScalarLaw:
    prior: NetworkScalarPrior
    mean: float | None
    sigma: float | None
    version: str = "network-scalar-law-v1"


@dataclass(frozen=True, slots=True)
class NetworkScalarValue:
    role: str
    law: NetworkScalarLaw
    quantile: float | None
    unrounded: float
    value: float
    version: str = "network-scalar-value-v1"


def resolve_scalar_law(
    prior: NetworkScalarPrior, rng: np.random.Generator, capture: RawDrawLedger, role: str
) -> NetworkScalarLaw:
    if type(prior) is not NetworkScalarPrior:
        raise ValueError("scalar sampling requires a frozen prior")
    if prior.family not in _TNLU:
        return NetworkScalarLaw(prior, None, None)

    def uniform(name: str, low: float, high: float) -> float:
        return float(
            capture.draw(
                rng,
                owner=("network-hyperprior", role),
                role=name,
                distribution="uniform",
                shape=(),
                parameters=(("low", low), ("high", high)),
            )
        )

    mu = math.exp(uniform("log-mean", math.log(prior.low), math.log(prior.high)))
    sigma = (
        math.exp(uniform("log-sigma", math.log(prior.low), math.log(prior.high)))
        if prior.family == "p1-paper-tnlu-v1"
        else mu * math.exp(uniform("log-relative-sigma", math.log(0.01), 0.0))
    )
    if not math.isfinite(mu) or not math.isfinite(sigma) or sigma <= 0:
        raise ValueError("nonfinite or zero hyperlaw scale; no retry")
    return NetworkScalarLaw(prior, mu, sigma)


def realize_scalar(
    law: NetworkScalarLaw, rng: np.random.Generator, capture: RawDrawLedger, role: str
) -> NetworkScalarValue:
    prior = law.prior
    quantile = None
    if prior.family == "fixed":
        raw = prior.low
    elif prior.family in ("choice", "integer-uniform"):
        count = len(prior.choices) if prior.family == "choice" else int(prior.high - prior.low) + 1
        slot = int(
            capture.draw(
                rng,
                owner=("network-hyperprior", role),
                role="slot",
                distribution="integers",
                shape=(),
                parameters=(("high", count),),
            )
        )
        raw = prior.choices[slot] if prior.family == "choice" else prior.low + slot
    else:
        quantile = float(
            capture.draw(
                rng,
                owner=("network-hyperprior", role),
                role="quantile",
                distribution="random",
                shape=(),
            )
        )
        if prior.family in _TNLU:
            assert law.mean is not None
            assert law.sigma is not None
            normal = NormalDist()
            lower = normal.cdf(-law.mean / law.sigma)
            upper = (
                1.0
                if prior.family == "p1-paper-tnlu-v1"
                else normal.cdf((1e6 - law.mean) / law.sigma)
            )
            probability = lower + quantile * (upper - lower)
            if not upper > lower or not 0 < probability < 1:
                raise ValueError("collapsed or endpoint truncated-normal quantile; no clipping")
            raw = law.mean + law.sigma * normal.inv_cdf(probability)
            if raw < 0 or prior.family == "tabicl-9530bd6-tnlu-v1" and raw > 1e6:
                raise ValueError("rounded arithmetic outside source truncation support")
        elif prior.family == "log-uniform":
            raw = math.exp(
                math.log(prior.low) + quantile * (math.log(prior.high) - math.log(prior.low))
            )
        else:
            raw = prior.low + quantile * (prior.high - prior.low)
    if not math.isfinite(raw):
        raise ValueError("nonfinite scalar realization; no retry")
    value = (float(round(raw)) if prior.rounded else raw) + prior.offset
    if not math.isfinite(value):
        raise ValueError("nonfinite offset scalar realization; no retry")
    return NetworkScalarValue(role, law, quantile, float(raw), float(value))
