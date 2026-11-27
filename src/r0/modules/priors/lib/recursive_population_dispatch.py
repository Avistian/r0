"""Dispatch retained per-row populations without reducing their lifecycle envelope."""

from typing import Any

from r0.modules.priors.lib.causal_limits import CausalLimits
from r0.modules.priors.lib.recursive_population_intervention import RecursivePopulationIntervention
from r0.modules.priors.lib.recursive_population_records import RecursivePopulationRecipe
from r0.modules.priors.lib.training_task import TrainingTask


def population_recipe_from_dict(
    fields: dict[str, Any], *, limits: CausalLimits
) -> RecursivePopulationRecipe | RecursivePopulationIntervention | None:
    version = fields.get("version")
    if (
        "parent_rows" in fields
        and "observed_rows" in fields
        and fields.get("version")
        not in ("recursive-population-v1", "recursive-population-intervention-v1")
    ):
        raise ValueError("population archive requires its explicit version")
    if version == "recursive-population-intervention-v1":
        from r0.modules.priors.lib.recursive_population_decoding import (
            recursive_population_intervention_from_dict,
        )

        return recursive_population_intervention_from_dict(fields, limits=limits)
    if version == "recursive-population-v1":
        from r0.modules.priors.lib.recursive_population_decoding import (
            recursive_population_from_dict,
        )

        return recursive_population_from_dict(fields, limits=limits)
    return None


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
