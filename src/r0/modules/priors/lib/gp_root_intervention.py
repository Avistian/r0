"""GP-root interventions retain source lineage and fixed independent target fitting."""

from dataclasses import asdict, dataclass

from r0.modules.priors.lib.causal_intervention_records import CausalInterventionRecipe
from r0.modules.priors.lib.causal_limits import CausalLimits
from r0.modules.priors.lib.gp_column_population import GPColumnPopulation, realize_gp_columns
from r0.modules.priors.lib.gp_population_metadata_cost import retained_cost
from r0.modules.priors.lib.gp_root_calibration import GPRootCalibration
from r0.modules.priors.lib.gp_root_dags import GPRootDAGRecipe
from r0.modules.priors.lib.gp_root_query import GPRootCalibrationQuery

type GPRootInterventionSource = GPRootDAGRecipe | GPRootCalibration | GPRootCalibrationQuery


@dataclass(frozen=True, slots=True)
class GPRootIntervention:
    base: GPRootInterventionSource
    change: CausalInterventionRecipe
    population: GPColumnPopulation
    limits: CausalLimits
    version: str = "gp-root-intervention-v1"

    def __post_init__(self) -> None:
        if (
            type(self.base) not in (GPRootDAGRecipe, GPRootCalibration, GPRootCalibrationQuery)
            or type(self.change) is not CausalInterventionRecipe
            or type(self.population) is not GPColumnPopulation
            or type(self.limits) is not CausalLimits
            or self.version != "gp-root-intervention-v1"
        ):
            raise ValueError("GP intervention requires complete source, change and population")

    @property
    def original(self) -> GPRootDAGRecipe:
        return self.base if isinstance(self.base, GPRootDAGRecipe) else self.base.original

    @property
    def task(self):
        return self.original.task

    def validate(self) -> None:
        from r0.modules.priors.lib.gp_root_intervention_decoding import (
            gp_root_intervention_from_dict,
        )

        if type(self.limits) is not CausalLimits:
            raise ValueError("GP intervention requires explicit limits")
        retained_cost(self, self.limits)
        gp_root_intervention_from_dict(asdict(self), limits=self.limits)


def intervene_gp_root_dag(
    base: GPRootInterventionSource,
    change: CausalInterventionRecipe,
    *,
    limits: CausalLimits | None = None,
) -> GPRootIntervention:
    from r0.modules.priors.lib.gp_root_intervention_decoding import (
        decode_gp_intervention_source,
    )
    from r0.modules.priors.lib.gp_root_intervention_metadata import intervention_plan

    if type(base) not in (GPRootDAGRecipe, GPRootCalibration, GPRootCalibrationQuery):
        raise ValueError("GP intervention requires a complete original GP-root source")
    if type(change) is not CausalInterventionRecipe:
        raise ValueError("GP intervention requires an explicit causal change")
    cap = base.limits if limits is None else limits
    if type(cap) is not CausalLimits:
        raise ValueError("GP intervention requires explicit caller limits")
    retained_cost((base, change), cap)
    metadata, _, seed, task_id = intervention_plan(asdict(base), asdict(change), limits=cap)
    valid = decode_gp_intervention_source(asdict(base), limits=cap)
    current = valid if isinstance(valid, GPRootDAGRecipe) else valid.original
    population = (
        current.source
        if change.mode == "matched"
        else realize_gp_columns(metadata.request, seed=seed, task_id=task_id)
    )
    result = GPRootIntervention(valid, change, population, metadata.limits)
    result.validate()
    return result
