"""Dispatch retained per-row populations without reducing their lifecycle envelope."""

from r0.modules.priors.lib.recursive_population_intervention import RecursivePopulationIntervention
from r0.modules.priors.lib.recursive_population_records import RecursivePopulationRecipe
from r0.modules.priors.lib.training_task import TrainingTask


def generate_population_recipe(
    recipe: RecursivePopulationRecipe | RecursivePopulationIntervention,
    *,
    seed: int,
    task_id: int,
    device: str,
) -> TrainingTask:
    if isinstance(recipe, RecursivePopulationIntervention):
        from r0.modules.priors.lib.recursive_population_intervention import (
            generate_population_intervention,
        )

        return generate_population_intervention(recipe, seed=seed, task_id=task_id, device=device)
    from r0.modules.priors.lib.recursive_population_output import generate_recursive_population

    return generate_recursive_population(recipe, seed=seed, task_id=task_id, device=device)
