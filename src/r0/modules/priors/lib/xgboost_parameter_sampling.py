"""Retained depth/count laws for actual XGBoost plans, separate from fitting."""

import math
from dataclasses import dataclass, replace

import numpy as np

from r0.modules.priors.lib.activation_sampling import finite_number
from r0.modules.priors.lib.causal_limits import CausalLimits
from r0.modules.priors.lib.fitted_parameter_laws import PINNED_SOURCE
from r0.modules.priors.lib.generator_snapshot import GeneratorSnapshot, snapshot_generator
from r0.modules.priors.lib.raw_draw_capture import RawDrawLedger
from r0.modules.priors.lib.raw_draw_records import RawDrawLimits, RawDrawRecord
from r0.modules.priors.lib.xgboost_fit_plan import XGBoostFitPlan

_PINNED = "tabicl-pinned-depth-count-v1"
_EXTENSION = "r0-capped-exponential-xgboost-v1"
_DEFAULT = CausalLimits()


@dataclass(frozen=True, slots=True)
class XGBoostParameterPrior:
    base: XGBoostFitPlan = XGBoostFitPlan()
    depth_rate: float = 0.5
    count_rate: float = 0.5
    depth_cap: int = 4
    count_cap: int = 4
    profile: str = _PINNED
    raw_limits: RawDrawLimits = RawDrawLimits()
    version: str = "xgboost-parameter-prior-v1"

    def __post_init__(self) -> None:
        if (
            self.version != "xgboost-parameter-prior-v1"
            or self.profile not in (_PINNED, _EXTENSION)
            or type(self.base) is not XGBoostFitPlan
            or type(self.raw_limits) is not RawDrawLimits
            or type(self.depth_cap) is not int
            or not 2 <= self.depth_cap <= 8
            or type(self.count_cap) is not int
            or not 1 <= self.count_cap <= 32
            or any(
                not finite_number(rate) or rate <= 0 or not math.isfinite(1 / rate)
                for rate in (self.depth_rate, self.count_rate)
            )
        ):
            raise ValueError("invalid XGBoost parameter prior")
        if self.profile == _PINNED and (
            (self.depth_rate, self.count_rate, self.depth_cap, self.count_cap) != (0.5, 0.5, 4, 4)
            or self.base.objective != "reg:squarederror"
        ):
            raise ValueError("pinned XGBoost law requires source rates/caps and regression")

    def admit(self, limits: CausalLimits) -> None:
        if type(limits) is not CausalLimits:
            raise ValueError("XGBoost sampling requires frozen caller limits")
        worst = replace(self.base, depth=self.depth_cap, estimators=self.count_cap)
        fit_cost = worst.admit(limits, self.raw_limits)
        # Two scalar draws, snapshots and Python/JSON replay copies coexist with fit.
        metadata = 8 * self.raw_limits.state_bytes + 4096
        if self.raw_limits.draws < 2 or self.raw_limits.numbers < 2:
            raise ValueError("XGBoost parameter raw count limit exceeded")
        if metadata > self.raw_limits.working_bytes or fit_cost + metadata > limits.working_bytes:
            raise ValueError("XGBoost parameter and fit working limit exceeded")


def _realize(
    prior: XGBoostParameterPrior, rng: np.random.Generator, ledger: RawDrawLedger
) -> XGBoostFitPlan:
    values: list[int] = []
    for name, rate, offset, cap in (
        ("depth", prior.depth_rate, 2, prior.depth_cap),
        ("estimators", prior.count_rate, 1, prior.count_cap),
    ):
        raw = float(
            ledger.draw(
                rng,
                owner=("xgboost-parameters",),
                role=name,
                distribution="exponential",
                shape=(),
                parameters=(("scale", 1 / rate),),
            ).item()
        )
        if not math.isfinite(raw) or raw < 0:
            raise ValueError("invalid exponential XGBoost parameter draw; not retried")
        values.append(cap if raw >= cap - offset else offset + math.floor(raw))
    return replace(prior.base, depth=values[0], estimators=values[1])


@dataclass(frozen=True, slots=True)
class SampledXGBoostPlan:
    prior: XGBoostParameterPrior
    plan: XGBoostFitPlan
    initial_rng: GeneratorSnapshot
    final_rng: GeneratorSnapshot
    raw: tuple[RawDrawRecord, ...]
    source: str
    version: str = "sampled-xgboost-plan-v1"

    def __post_init__(self) -> None:
        if (
            self.version != "sampled-xgboost-plan-v1"
            or type(self.prior) is not XGBoostParameterPrior
            or type(self.plan) is not XGBoostFitPlan
            or type(self.initial_rng) is not GeneratorSnapshot
            or type(self.final_rng) is not GeneratorSnapshot
            or type(self.raw) is not tuple
            or len(self.raw) != 2
            or any(type(r) is not RawDrawRecord for r in self.raw)
        ):
            raise ValueError("invalid sampled XGBoost parameter record")
        expected_source = PINNED_SOURCE if self.prior.profile == _PINNED else _EXTENSION
        if self.source != expected_source:
            raise ValueError("XGBoost parameter source differs")
        rng = self.initial_rng.restore()
        ledger = RawDrawLedger(limits=self.prior.raw_limits)
        with ledger.transactions({"parameters": rng}) as (streams, capture):
            active = streams["parameters"]
            plan = _realize(self.prior, active, capture)
            if (
                plan != self.plan
                or capture.freeze() != self.raw
                or snapshot_generator(active) != self.final_rng
            ):
                raise ValueError("XGBoost parameter replay differs")


def sample_xgboost_plan(
    prior: XGBoostParameterPrior, *, rng: np.random.Generator, limits: CausalLimits = _DEFAULT
) -> SampledXGBoostPlan:
    if type(prior) is not XGBoostParameterPrior:
        raise ValueError("XGBoost sampling requires a frozen parameter prior")
    prior.admit(limits)
    ledger = RawDrawLedger(limits=prior.raw_limits)
    with ledger.transactions({"parameters": rng}) as (streams, capture):
        active = streams["parameters"]
        initial = snapshot_generator(active, byte_limit=prior.raw_limits.state_bytes)
        plan = _realize(prior, active, capture)
        return SampledXGBoostPlan(
            prior,
            plan,
            initial,
            snapshot_generator(active, byte_limit=prior.raw_limits.state_bytes),
            capture.freeze(),
            PINNED_SOURCE if prior.profile == _PINNED else _EXTENSION,
        )
