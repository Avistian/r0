"""Fixed independent calibration at original-row reference points, never a grid fit."""

from dataclasses import dataclass

from r0.modules.priors.lib.calibrated_population_intervention import (
    CalibratedPopulationIntervention,
)
from r0.modules.priors.lib.calibrated_population_query import CalibratedPopulationQuery
from r0.modules.priors.lib.causal_intervention_execution import InterventionExecution
from r0.modules.priors.lib.causal_intervention_records import CausalInterventionRecipe
from r0.modules.priors.lib.functional_causal import FunctionalCausalRecipe
from r0.modules.priors.lib.functional_causal_evaluation import functional_rng
from r0.modules.priors.lib.population_calibrated_output import apply_population_calibration
from r0.modules.priors.lib.population_reference import PopulationGraphReference
from r0.modules.priors.lib.population_target_calibration import CalibratedPopulationGraph
from r0.modules.priors.lib.training_task import TrainingTask

type CalibratedReferenceSource = (
    CalibratedPopulationGraph | CalibratedPopulationQuery | CalibratedPopulationIntervention
)
CALIBRATED_REFERENCE_TYPES = (
    CalibratedPopulationGraph,
    CalibratedPopulationQuery,
    CalibratedPopulationIntervention,
)


@dataclass(frozen=True, slots=True)
class CalibratedPopulationReference(PopulationGraphReference):
    raw_scores: tuple[tuple[float, ...], ...] = ()
    calibration_fingerprint: str = ""
    version: str = "calibrated-population-graph-reference-v1"


def calibrated_reference_outputs(
    source: CalibratedReferenceSource, raw: TrainingTask, change: CausalInterventionRecipe
):
    baseline = source.base if isinstance(source, CalibratedPopulationIntervention) else source
    seed, task_id = baseline.original.seed, baseline.original.task_id
    execution = InterventionExecution(change)
    return apply_population_calibration(
        raw,
        source.calibration,
        adapter=source.task
        if isinstance(baseline.original.request.base, FunctionalCausalRecipe)
        else None,
        rng=lambda key: execution.rng(lambda *keys: functional_rng(seed, task_id, *keys), key),
    )
