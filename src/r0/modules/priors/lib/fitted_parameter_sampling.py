"""One owned parameter stream, exact raw replay and atomic caller RNG commits."""

import math
from dataclasses import dataclass, replace
from typing import Any

import numpy as np

from r0.modules.priors.lib.causal_limits import CausalLimits
from r0.modules.priors.lib.fitted_parameter_laws import PINNED_SOURCE, FittedParameterLaw
from r0.modules.priors.lib.fitted_parameter_prior import FittedTreeParameterPrior
from r0.modules.priors.lib.fitted_tree_plan import FittedTreePlan
from r0.modules.priors.lib.generator_snapshot import GeneratorSnapshot, snapshot_generator
from r0.modules.priors.lib.raw_draw_capture import RawDrawLedger
from r0.modules.priors.lib.raw_draw_records import RawDrawRecord

_DEFAULT_LIMITS = CausalLimits()


def _realize(
    prior: FittedTreeParameterPrior, rng: np.random.Generator, capture: RawDrawLedger
) -> FittedTreePlan:
    def draw(role: str, law: str, parameters: tuple[tuple[str, Any], ...] = ()) -> Any:
        return capture.draw(
            rng,
            owner=("fitted-tree-parameters",),
            role=role,
            distribution=law,
            shape=(),
            parameters=parameters,
        ).item()

    def value(name: str, law: FittedParameterLaw) -> Any:
        if law.kind == "fixed":
            return law.values[0]
        if law.kind == "categorical":
            if law.probabilities:
                index = draw(
                    name,
                    "choice",
                    (("count", len(law.values)), ("probabilities", law.probabilities)),
                )
            else:
                index = draw(name, "integers", (("high", len(law.values)),))
            return law.values[int(index)]
        if law.kind == "integer-uniform":
            return int(law.low) + int(draw(name, "integers", (("high", law.high - law.low + 1),)))
        if law.kind == "capped-exponential":
            raw = float(draw(name, "exponential", (("scale", 1 / law.rate),)))
            if not math.isfinite(raw) or raw < 0:
                raise ValueError("nonfinite exponential parameter draw; not retried")
            # Retain the capped tail mass without converting an enormous float to int.
            return law.cap if raw >= law.cap - law.offset else law.offset + math.floor(raw)
        if law.low == law.high:
            return law.low
        low, high = (
            (math.log(law.low), math.log(law.high))
            if law.kind == "log-uniform"
            else (law.low, law.high)
        )
        raw = float(draw(name, "uniform", (("low", low), ("high", high))))
        return math.exp(raw) if law.kind == "log-uniform" else raw

    return replace(prior.base, **{name: value(name, law) for name, law in prior.effective_laws})


@dataclass(frozen=True, slots=True)
class SampledFittedTreePlan:
    prior: FittedTreeParameterPrior
    plan: FittedTreePlan
    initial_rng: GeneratorSnapshot
    final_rng: GeneratorSnapshot
    raw: tuple[RawDrawRecord, ...]
    source: str
    unused_parameters: tuple[str, ...] = ()
    backend_semantics: str = "sklearn-independent-output-r0-v1"
    lifecycle: str = "independent-calibration-r0-v1"
    version: str = "sampled-fitted-tree-plan-v1"

    def __post_init__(self) -> None:
        if (
            self.version != "sampled-fitted-tree-plan-v1"
            or type(self.prior) is not FittedTreeParameterPrior
            or type(self.plan) is not FittedTreePlan
            or type(self.initial_rng) is not GeneratorSnapshot
            or type(self.final_rng) is not GeneratorSnapshot
            or type(self.raw) is not tuple
            or any(type(r) is not RawDrawRecord for r in self.raw)
            or self.backend_semantics != "sklearn-independent-output-r0-v1"
            or self.lifecycle != "independent-calibration-r0-v1"
        ):
            raise ValueError("invalid sampled fitted parameter provenance")
        pinned = self.prior.profile == "tabicl-pinned-depth-count-v1"
        if self.source != (
            PINNED_SOURCE if pinned else "r0-explicit-parameter-laws-v1"
        ) or self.unused_parameters != (
            ("estimators",) if pinned and self.plan.family == "dt" else ()
        ):
            raise ValueError("parameter source or unused backend choice label differs")
        rng = self.initial_rng.restore()
        ledger = RawDrawLedger(limits=self.prior.raw_limits)
        with ledger.transactions({"parameters": rng}) as (streams, capture):
            plan = _realize(self.prior, streams["parameters"], capture)
            if (
                plan != self.plan
                or capture.freeze() != self.raw
                or snapshot_generator(streams["parameters"]) != self.final_rng
            ):
                raise ValueError("sampled fit parameters differ from raw custody replay")


def sample_fitted_tree_plan(
    prior: FittedTreeParameterPrior,
    *,
    rng: np.random.Generator,
    limits: CausalLimits = _DEFAULT_LIMITS,
) -> SampledFittedTreePlan:
    if type(prior) is not FittedTreeParameterPrior or type(limits) is not CausalLimits:
        raise ValueError("fitted parameter sampling requires frozen prior and caller limits")
    prior.admit(limits)
    ledger = RawDrawLedger(limits=prior.raw_limits)
    with ledger.transactions({"parameters": rng}) as (streams, capture):
        active = streams["parameters"]
        initial = snapshot_generator(active, byte_limit=prior.raw_limits.state_bytes)
        plan = _realize(prior, active, capture)
        pinned = prior.profile == "tabicl-pinned-depth-count-v1"
        return SampledFittedTreePlan(
            prior,
            plan,
            initial,
            snapshot_generator(active, byte_limit=prior.raw_limits.state_bytes),
            capture.freeze(),
            PINNED_SOURCE if pinned else "r0-explicit-parameter-laws-v1",
            ("estimators",) if pinned and plan.family == "dt" else (),
        )
