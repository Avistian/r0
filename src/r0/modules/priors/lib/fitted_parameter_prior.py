"""Admit complete fitted parameter option spaces before sampling or fitting."""

from dataclasses import dataclass, replace
from itertools import product
from typing import Any

from r0.modules.priors.lib.activation_sampling import finite_number
from r0.modules.priors.lib.causal_limits import CausalLimits
from r0.modules.priors.lib.fitted_parameter_laws import (
    PINNED_COUNT,
    PINNED_DEPTH,
    FittedParameterLaw,
)
from r0.modules.priors.lib.fitted_tree_plan import FittedTreePlan
from r0.modules.priors.lib.raw_draw_records import RawDrawLimits

PARAMETERS = (
    "family",
    "depth",
    "estimators",
    "rows",
    "classes",
    "splitter",
    "classification_rule",
    "learning_rate",
)


@dataclass(frozen=True, slots=True)
class FittedTreeParameterPrior:
    base: FittedTreePlan = FittedTreePlan()
    laws: tuple[tuple[str, FittedParameterLaw], ...] = ()
    profile: str = "r0-explicit-parameter-laws-v1"
    raw_limits: RawDrawLimits = RawDrawLimits()
    version: str = "fitted-tree-parameter-prior-v1"

    def __post_init__(self) -> None:
        if (
            self.version != "fitted-tree-parameter-prior-v1"
            or self.profile not in ("r0-explicit-parameter-laws-v1", "tabicl-pinned-depth-count-v1")
            or type(self.base) is not FittedTreePlan
            or type(self.raw_limits) is not RawDrawLimits
            or type(self.laws) is not tuple
            or len(self.laws) > len(PARAMETERS)
            or any(
                type(v) is not tuple
                or len(v) != 2
                or v[0] not in PARAMETERS
                or type(v[1]) is not FittedParameterLaw
                for v in self.laws
            )
            or len({v[0] for v in self.laws}) != len(self.laws)
        ):
            raise ValueError("invalid fitted parameter prior or repeated parameter")
        if self.profile == "tabicl-pinned-depth-count-v1" and any(
            name in ("depth", "estimators") for name, _ in self.laws
        ):
            raise ValueError("pinned depth/count laws cannot be overridden; use explicit profile")
        self.options()

    @property
    def effective_laws(self) -> tuple[tuple[str, FittedParameterLaw], ...]:
        return (
            (("depth", PINNED_DEPTH), ("estimators", PINNED_COUNT))
            if self.profile == "tabicl-pinned-depth-count-v1"
            else ()
        ) + self.laws

    def options(self) -> tuple[FittedTreePlan, ...]:
        laws = dict(self.effective_laws)

        def choices(name: str) -> tuple[Any, ...]:
            return laws[name].extrema if name in laws else (getattr(self.base, name),)

        for name, allowed in (
            ("family", ("dt", "extra_trees", "rf", "gbdt")),
            ("splitter", ("best", "random")),
            ("classification_rule", ("probability_mean", "majority")),
        ):
            values = choices(name)
            if any(v not in allowed for v in values) or len(set(values)) != len(values):
                raise ValueError("backend parameter options must be valid unique names")
        families = choices("family")
        for name, law in self.effective_laws:
            if name not in ("family", "splitter", "classification_rule") and any(
                not finite_number(value) for value in law.extrema
            ):
                raise ValueError("numeric parameter laws require finite numeric choices")
            if name in ("family", "splitter", "classification_rule") and law.kind not in (
                "fixed",
                "categorical",
            ):
                raise ValueError("named backend parameters require fixed or categorical laws")
            if name in ("depth", "estimators", "rows", "classes") and (
                law.kind in ("uniform", "log-uniform")
                or any(type(v) is not int for v in law.extrema)
            ):
                raise ValueError("integer fit parameters require integer laws and values")
            if name == "learning_rate" and law.kind in ("integer-uniform", "capped-exponential"):
                raise ValueError("learning rate requires a continuous or explicit-value law")
            unused = (
                name == "learning_rate"
                and any(f != "gbdt" for f in families)
                or name == "classes"
                and self.base.task != "classification"
                or name == "splitter"
                and any(f != "dt" for f in families)
                or name == "classification_rule"
                and (self.base.task != "classification" or "gbdt" in families)
                or name == "estimators"
                and "dt" in families
                and self.profile != "tabicl-pinned-depth-count-v1"
            )
            if unused:
                raise ValueError("explicit law varies an unused backend/task parameter")
        # Only categorical backend choices need Cartesian enumeration. Numerical
        # upper/lower extrema establish monotone bounded fit envelopes.
        numeric = ("depth", "estimators", "rows", "classes", "learning_rate")
        low = {name: min(choices(name)) for name in numeric}
        high = {name: max(choices(name)) for name in numeric}
        result: list[FittedTreePlan] = []
        for family, splitter, rule in product(
            families, choices("splitter"), choices("classification_rule")
        ):
            for extremes in (low, high):
                result.append(
                    replace(
                        self.base,
                        family=family,
                        splitter=splitter,
                        classification_rule=rule,
                        **extremes,
                    )
                )
        return tuple(result)

    def admit(self, limits: CausalLimits) -> None:
        for plan in self.options():
            plan.admit(limits)
        draws = sum(law.stochastic for _, law in self.effective_laws)
        raw = self.raw_limits
        # Scalar draws, two snapshots and bounded per-record Python/JSON overhead.
        metadata = 2 * raw.state_bytes + draws * (1024 + 2 * raw.state_bytes)
        if draws > raw.draws or draws > raw.numbers or metadata > raw.working_bytes:
            raise ValueError("fitted parameter raw custody budget exceeded before draws")
        if metadata > limits.working_bytes:
            raise ValueError("fitted parameter metadata exceeds caller memory cap")
