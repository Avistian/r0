"""Counterfactual row populations reuse baseline independent calibration without refitting."""

from dataclasses import asdict, dataclass

from r0.modules.priors.lib.causal_intervention_records import CausalInterventionRecipe
from r0.modules.priors.lib.causal_limits import CausalLimits
from r0.modules.priors.lib.exact_frozen_record import exact_frozen_equal
from r0.modules.priors.lib.joint_function_admission import function_record_cost
from r0.modules.priors.lib.population_target_calibration import CalibratedPopulationGraph
from r0.modules.priors.lib.recursive_population_graph_intervention import (
    RecursivePopulationGraphIntervention,
    intervene_recursive_population_graph,
)


@dataclass(frozen=True, slots=True)
class CalibratedPopulationIntervention:
    base: CalibratedPopulationGraph
    original: RecursivePopulationGraphIntervention
    version: str = "calibrated-population-intervention-v1"

    @property
    def task(self):
        return self.base.task

    @property
    def limits(self):
        return self.base.limits

    @property
    def calibration(self):
        return self.base.calibration

    def validate(self, *, limits: CausalLimits | None = None) -> None:
        from r0.modules.priors.lib.calibrated_population_intervention_decoding import (
            preflight_calibrated_intervention,
        )

        if (
            type(self.base) is not CalibratedPopulationGraph
            or type(self.original) is not RecursivePopulationGraphIntervention
            or self.version != "calibrated-population-intervention-v1"
            or not exact_frozen_equal(self.original.base, self.base.original)
        ):
            raise ValueError("calibrated intervention must retain its exact original population")
        cap = limits or self.limits
        function_record_cost(self, cap)
        preflight_calibrated_intervention(asdict(self), limits=cap)
        self.base.validate()
        self.original.validate_replay(limits=cap)


def intervene_calibrated_population(
    base: CalibratedPopulationGraph, change: CausalInterventionRecipe
) -> CalibratedPopulationIntervention:
    from r0.modules.priors.lib.calibrated_population_intervention_decoding import (
        admit_calibrated_change,
    )

    if type(base) is not CalibratedPopulationGraph or type(change) is not CausalInterventionRecipe:
        raise ValueError("calibrated intervention requires a retained baseline and causal change")
    if not exact_frozen_equal(change.base, base.original.request.base):
        raise ValueError("calibrated intervention change must retain the original authored DAG")
    admit_calibrated_change(base, change)
    base.validate()
    original = intervene_recursive_population_graph(base.original, change)
    result = CalibratedPopulationIntervention(base, original)
    result.validate()
    return result
