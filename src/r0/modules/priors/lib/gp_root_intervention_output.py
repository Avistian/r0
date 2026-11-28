"""Execute surgical changes on retained GP roots, preserving independent target fits."""

import json
from dataclasses import asdict, replace

from r0.modules.priors.lib.causal_intervention_execution import intervention_execution
from r0.modules.priors.lib.causal_intervention_generation import generate_intervened
from r0.modules.priors.lib.functional_causal import FunctionalCausalRecipe
from r0.modules.priors.lib.functional_causal_evaluation import functional_rng
from r0.modules.priors.lib.gp_root_dags import GPRootDAGRecipe
from r0.modules.priors.lib.gp_root_intervention import GPRootIntervention
from r0.modules.priors.lib.gp_root_intervention_decoding import gp_root_intervention_from_dict
from r0.modules.priors.lib.representatives import encode, record_fingerprint
from r0.modules.priors.lib.targets import adapt_targets
from r0.modules.priors.lib.training_task import TrainingTask


def generate_gp_root_intervention(
    recipe: GPRootIntervention, *, seed: int, task_id: int, device: str = "cpu"
) -> TrainingTask:
    if (
        any(type(value) is not int for value in (seed, task_id))
        or (seed, task_id) != (recipe.original.source.seed, recipe.original.source.task_id)
        or device != "cpu"
    ):
        raise ValueError("GP intervention requires exact original identity and CPU replay")
    valid = gp_root_intervention_from_dict(asdict(recipe), limits=recipe.limits)
    context = replace(valid.original, source=valid.population).context()
    execution = intervention_execution(valid.change)
    task = generate_intervened(
        valid.change, seed=seed, task_id=task_id, execution=execution, root_populations=context
    )
    if not isinstance(valid.base, GPRootDAGRecipe):
        scores = valid.base.calibration.apply(task.scores)
        if isinstance(valid.original.base, FunctionalCausalRecipe):
            targets, probabilities, classes = adapt_targets(
                valid.task,
                scores,
                noise_rng=execution.rng(
                    lambda *keys: functional_rng(seed, task_id, *keys), "noise"
                ),
                label_rng=execution.rng(
                    lambda *keys: functional_rng(seed, task_id, *keys), "labels"
                ),
            )
        else:
            if task.classes:
                raise ValueError("direct categorical GP targets cannot receive numeric calibration")
            targets, probabilities, classes = scores[:, 0], None, task.classes
        task = replace(
            task, scores=scores, targets=targets, probabilities=probabilities, classes=classes
        )
    config = asdict(valid)
    record = json.loads(task.recipe_json)
    record["recipe"] = config
    record["execution"]["implementation"] = valid.version
    identity = record_fingerprint((config, seed, task_id))
    return replace(
        task,
        source_task_id=identity,
        episode_id=record_fingerprint((identity, valid.task.split, valid.task.context_rows)),
        recipe_json=encode(record),
        mechanism_json=encode(
            {
                "family": "gp_root_intervention",
                "version": valid.version,
                "original": json.loads(task.mechanism_json),
                "population_id": valid.population.population_id,
                "original_population_id": valid.original.source.population_id,
                "bindings": [asdict(binding) for binding in valid.original.bindings],
                "calibration": None
                if isinstance(valid.base, GPRootDAGRecipe)
                else asdict(valid.base.calibration),
                "query_fit": False,
            }
        ),
    )
