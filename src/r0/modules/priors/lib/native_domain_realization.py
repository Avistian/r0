"""Retain raw native frequency and threshold draws; no empirical class rebalance."""

from statistics import NormalDist
from typing import Literal, cast

import numpy as np

from r0.modules.priors.lib.native_causal import FrequencyVariableDomain, VariableDomain
from r0.modules.priors.lib.native_domain_prior import AsymmetricNativeDomainPlan, NativeDomainPlan
from r0.modules.priors.lib.raw_draw_capture import RawDrawLedger, recorded_draw


def materialize_native_domain(
    plan: NativeDomainPlan,
    rng: np.random.Generator,
    capture: RawDrawLedger,
) -> VariableDomain:
    if plan.kind == "continuous":
        return VariableDomain()
    k = plan.cardinality
    if plan.frequency_law == "uniform":
        weights = np.ones(k, dtype=np.float64)
    elif plan.frequency_law == "asymmetric-dirichlet":
        assert isinstance(plan, AsymmetricNativeDomainPlan)
        weights = np.array(
            [
                recorded_draw(
                    rng,
                    capture,
                    owner=plan.operation,
                    role=f"frequency:{i}",
                    distribution="gamma",
                    shape=(1,),
                    parameters=(("concentration", alpha),),
                )[0]
                for i, alpha in enumerate(plan.concentrations)
            ],
            dtype=np.float64,
        )
    else:
        raw = recorded_draw(
            rng,
            capture,
            owner=plan.operation,
            role="frequency",
            distribution="gamma" if plan.frequency_law == "dirichlet" else "normal",
            shape=(k,),
            parameters=(("concentration", plan.concentration),)
            if plan.frequency_law == "dirichlet"
            else (),
        )
        weights = (
            raw
            if plan.frequency_law == "dirichlet"
            else np.exp((raw - raw.max()) * plan.concentration)
        )
    total = weights.sum()
    if not np.isfinite(total) or total <= 0 or np.any(weights <= 0):
        raise ValueError("nonfinite or underflowed native frequencies; no retry")
    probabilities = weights / total
    if not np.isfinite(probabilities).all() or np.any(probabilities <= 0):
        raise ValueError("nonfinite or underflowed native frequencies; no retry")
    thresholds: tuple[float, ...] = ()
    if plan.emission == "threshold":
        cdf = probabilities.cumsum()[:-1]
        if plan.threshold_law == "sorted-normal":
            cuts = np.sort(
                recorded_draw(
                    rng,
                    capture,
                    owner=plan.operation,
                    role="thresholds",
                    distribution="normal",
                    shape=(k - 1,),
                )
            )
        elif plan.threshold_law == "uniform-quantile":
            cuts = 2 * cdf - 1
        else:
            if np.any(cdf <= 0) or np.any(cdf >= 1):
                raise ValueError("native reference quantiles rounded to an endpoint; no retry")
            cuts = np.array([NormalDist().inv_cdf(float(p)) for p in cdf])
        thresholds = tuple(float(v) for v in cuts * plan.threshold_scale)
    return FrequencyVariableDomain(
        kind=cast(Literal["binary", "categorical"], plan.kind),
        vocabulary=tuple(f"category-{i}" for i in range(k)),
        emission=cast(Literal["softmax", "argmax", "threshold"], plan.emission),
        thresholds=thresholds,
        temperature=plan.temperature,
        frequencies=tuple(float(p) for p in probabilities),
    )
