"""One owned parameter stream, with effective rows bound to the selected lifecycle."""

from dataclasses import dataclass

import numpy as np

from r0.modules.priors.lib.causal_limits import CausalLimits
from r0.modules.priors.lib.fitted_parameter_laws import PINNED_SOURCE
from r0.modules.priors.lib.fitted_parameter_sampling import realize_fitted_parameter_plan
from r0.modules.priors.lib.fitted_population_fit import admit_population_fit
from r0.modules.priors.lib.fitted_population_plan import PopulationTreePlan
from r0.modules.priors.lib.generator_snapshot import GeneratorSnapshot, snapshot_generator
from r0.modules.priors.lib.population_parameter_prior import PopulationTreeParameterPrior
from r0.modules.priors.lib.raw_draw_capture import RawDrawLedger
from r0.modules.priors.lib.raw_draw_records import RawDrawRecord

_DEFAULT_LIMITS = CausalLimits()
_ROWS = {
    "support-only": "actual-episode-context-rows-v1",
    "batch-refit-reference": "actual-episode-batch-rows-v1",
    "mixed-independent": "independent-selected-fit-rows-v1",
}


@dataclass(frozen=True, slots=True)
class SampledPopulationTreePlan:
    prior: PopulationTreeParameterPrior
    plan: PopulationTreePlan
    initial_rng: GeneratorSnapshot
    final_rng: GeneratorSnapshot
    raw: tuple[RawDrawRecord, ...]
    parameter_source: str
    row_binding: str
    unused_parameters: tuple[str, ...] = ()
    backend_semantics: str = "sklearn-independent-output-r0-v1"
    lifecycle: str = "explicit-population-freeze-r0-v1"
    version: str = "sampled-population-tree-plan-v1"

    def __post_init__(self) -> None:
        if (
            self.version != "sampled-population-tree-plan-v1"
            or type(self.prior) is not PopulationTreeParameterPrior
            or type(self.plan) is not PopulationTreePlan
            or type(self.initial_rng) is not GeneratorSnapshot
            or type(self.final_rng) is not GeneratorSnapshot
            or type(self.raw) is not tuple
            or any(type(r) is not RawDrawRecord for r in self.raw)
            or self.backend_semantics != "sklearn-independent-output-r0-v1"
            or self.lifecycle != "explicit-population-freeze-r0-v1"
        ):
            raise ValueError("invalid sampled population parameter provenance")
        pinned = self.prior.profile == "tabicl-pinned-depth-count-v1"
        if (
            self.parameter_source != (PINNED_SOURCE if pinned else "r0-explicit-parameter-laws-v1")
            or self.row_binding != _ROWS[self.prior.population]
            or self.unused_parameters
            != (("estimators",) if pinned and self.plan.family == "dt" else ())
        ):
            raise ValueError("population parameter source, effective rows or unused labels differ")
        rng = self.initial_rng.restore()
        ledger = RawDrawLedger(limits=self.prior.raw_limits)
        with ledger.transactions({"parameters": rng}) as (streams, capture):
            plan = realize_fitted_parameter_plan(self.prior, streams["parameters"], capture)
            if (
                plan != self.plan
                or capture.freeze() != self.raw
                or snapshot_generator(streams["parameters"]) != self.final_rng
            ):
                raise ValueError("population sampled parameters differ from raw custody replay")


def sample_population_tree_plan(
    prior: PopulationTreeParameterPrior,
    *,
    rng: np.random.Generator,
    limits: CausalLimits = _DEFAULT_LIMITS,
) -> SampledPopulationTreePlan:
    if type(prior) is not PopulationTreeParameterPrior or type(limits) is not CausalLimits:
        raise ValueError("population sampling requires its frozen prior and caller limits")
    prior.admit(limits)
    for plan in prior.options():
        assert isinstance(plan, PopulationTreePlan)
        admit_population_fit(plan, limits)
    ledger = RawDrawLedger(limits=prior.raw_limits)
    with ledger.transactions({"parameters": rng}) as (streams, capture):
        active = streams["parameters"]
        initial = snapshot_generator(active, byte_limit=prior.raw_limits.state_bytes)
        plan = realize_fitted_parameter_plan(prior, active, capture)
        assert isinstance(plan, PopulationTreePlan)
        pinned = prior.profile == "tabicl-pinned-depth-count-v1"
        return SampledPopulationTreePlan(
            prior,
            plan,
            initial,
            snapshot_generator(active, byte_limit=prior.raw_limits.state_bytes),
            capture.freeze(),
            PINNED_SOURCE if pinned else "r0-explicit-parameter-laws-v1",
            _ROWS[prior.population],
            ("estimators",) if pinned and plan.family == "dt" else (),
        )
