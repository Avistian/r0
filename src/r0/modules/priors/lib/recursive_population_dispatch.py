"""Dispatch retained per-row populations without reducing their lifecycle envelope."""

from r0.modules.priors.lib.calibrated_population_intervention import (
    CalibratedPopulationIntervention,
)
from r0.modules.priors.lib.calibrated_population_query import CalibratedPopulationQuery
from r0.modules.priors.lib.gp_root_calibration import GPRootCalibration, generate_calibrated_gp_root
from r0.modules.priors.lib.gp_root_dags import GPRootDAGRecipe, generate_gp_root_dag
from r0.modules.priors.lib.gp_root_intervention import GPRootIntervention
from r0.modules.priors.lib.gp_root_intervention_output import generate_gp_root_intervention
from r0.modules.priors.lib.gp_root_query import GPRootCalibrationQuery
from r0.modules.priors.lib.population_recipe_decoding import (
    population_recipe_from_dict as population_recipe_from_dict,
)
from r0.modules.priors.lib.population_target_calibration import CalibratedPopulationGraph
from r0.modules.priors.lib.recursive_population_graph_intervention import (
    RecursivePopulationGraphIntervention,
)
from r0.modules.priors.lib.recursive_population_graph_records import RecursivePopulationGraph
from r0.modules.priors.lib.recursive_population_intervention import RecursivePopulationIntervention
from r0.modules.priors.lib.recursive_population_records import RecursivePopulationRecipe
from r0.modules.priors.lib.training_task import TrainingTask

type PopulationRecipe = (
    GPRootIntervention
    | GPRootCalibrationQuery
    | GPRootCalibration
    | GPRootDAGRecipe
    | CalibratedPopulationQuery
    | CalibratedPopulationGraph
    | CalibratedPopulationIntervention
    | RecursivePopulationRecipe
    | RecursivePopulationIntervention
    | RecursivePopulationGraph
    | RecursivePopulationGraphIntervention
)

POPULATION_RECIPE_TYPES = (
    GPRootIntervention,
    GPRootCalibrationQuery,
    GPRootCalibration,
    GPRootDAGRecipe,
    CalibratedPopulationQuery,
    CalibratedPopulationGraph,
    CalibratedPopulationIntervention,
    RecursivePopulationRecipe,
    RecursivePopulationIntervention,
    RecursivePopulationGraph,
    RecursivePopulationGraphIntervention,
)


def generate_population_recipe(
    recipe: PopulationRecipe,
    *,
    seed: int,
    task_id: int,
    device: str,
) -> TrainingTask:
    if isinstance(recipe, GPRootIntervention):
        return generate_gp_root_intervention(recipe, seed=seed, task_id=task_id, device=device)
    if isinstance(recipe, (GPRootCalibration, GPRootCalibrationQuery)):
        return generate_calibrated_gp_root(recipe, seed=seed, task_id=task_id, device=device)
    if isinstance(recipe, GPRootDAGRecipe):
        return generate_gp_root_dag(recipe, seed=seed, task_id=task_id, device=device)
    if isinstance(recipe, CalibratedPopulationQuery):
        from r0.modules.priors.lib.calibrated_query_output import generate_calibrated_query

        return generate_calibrated_query(recipe, seed=seed, task_id=task_id, device=device)
    if isinstance(recipe, CalibratedPopulationIntervention):
        from r0.modules.priors.lib.calibrated_population_intervention_output import (
            generate_calibrated_intervention,
        )

        return generate_calibrated_intervention(recipe, seed=seed, task_id=task_id, device=device)
    if isinstance(recipe, CalibratedPopulationGraph):
        from r0.modules.priors.lib.population_calibrated_output import (
            generate_calibrated_population,
        )

        return generate_calibrated_population(recipe, seed=seed, task_id=task_id, device=device)
    if isinstance(recipe, RecursivePopulationGraphIntervention):
        from r0.modules.priors.lib.recursive_population_graph_intervention import (
            generate_population_graph_intervention,
        )

        return generate_population_graph_intervention(
            recipe, seed=seed, task_id=task_id, device=device
        )
    if isinstance(recipe, RecursivePopulationGraph):
        from r0.modules.priors.lib.recursive_population_graph_output import (
            generate_recursive_population_graph,
        )

        return generate_recursive_population_graph(
            recipe, seed=seed, task_id=task_id, device=device
        )
    if isinstance(recipe, RecursivePopulationIntervention):
        from r0.modules.priors.lib.recursive_population_intervention import (
            generate_population_intervention,
        )

        return generate_population_intervention(recipe, seed=seed, task_id=task_id, device=device)
    from r0.modules.priors.lib.recursive_population_output import generate_recursive_population

    return generate_recursive_population(recipe, seed=seed, task_id=task_id, device=device)
