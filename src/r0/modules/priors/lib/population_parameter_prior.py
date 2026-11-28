"""Parameter laws composed with explicit fitting-population contracts."""

from dataclasses import dataclass

from r0.modules.priors.lib.fitted_parameter_prior import (
    FittedTreeParameterPrior,
    validate_fitted_parameter_prior,
)
from r0.modules.priors.lib.fitted_population_plan import PopulationTreePlan


@dataclass(frozen=True, slots=True)
class PopulationTreeParameterPrior(FittedTreeParameterPrior):
    base: PopulationTreePlan = PopulationTreePlan()
    population: str = "support-only"
    version: str = "population-tree-parameter-prior-v1"

    def __post_init__(self) -> None:
        validate_fitted_parameter_prior(
            self, PopulationTreePlan, "population-tree-parameter-prior-v1"
        )
        if self.population not in ("support-only", "batch-refit-reference", "mixed-independent"):
            raise ValueError("population parameter prior requires an explicit fitting lifecycle")
        if self.population != "mixed-independent" and any(name == "rows" for name, _ in self.laws):
            raise ValueError("support/batch row count is fixed by the episode; rows law is unused")
