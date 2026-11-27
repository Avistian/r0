"""Full population DAG provenance wraps shared numerical and typed task outputs."""

import json
from dataclasses import asdict, replace
from typing import Any

from r0.modules.priors.lib.recursive_population_graph_records import RecursivePopulationGraph
from r0.modules.priors.lib.recursive_population_graph_runtime import PopulationGraphEvaluation
from r0.modules.priors.lib.recursive_population_records import population_identity
from r0.modules.priors.lib.representatives import encode, record_fingerprint
from r0.modules.priors.lib.training_task import TrainingTask


def population_graph_task(
    recipe: RecursivePopulationGraph,
    evaluation: PopulationGraphEvaluation,
    *,
    config: dict[str, Any] | None = None,
    intervention: dict[str, Any] | None = None,
) -> TrainingTask:
    config = asdict(recipe) if config is None else config
    task = evaluation.task
    record = json.loads(task.recipe_json)
    record["recipe"] = config
    record["execution"]["implementation"] = config["version"]
    runtime_card = json.loads(task.mechanism_json)
    card = {
        "family": "recursive_population_graph",
        "version": config["version"],
        "lifecycle": "fresh-forest-per-row-per-replaced-node",
        "original_base_identity": record_fingerprint(asdict(recipe.request.base)),
        "original_base_role": (
            "unchanged counterfactual; declared replacements override selected mechanisms"
        ),
        "population_nodes": [
            {
                "node": b.node,
                "plan": asdict(b.sample.plan),
                "inputs_digest": record_fingerprint(b.inputs),
                "initial_draw_state": b.sample.initial.digest,
                "final_draw_state": b.sample.final.digest,
            }
            for b in recipe.bindings
        ],
        "source_scope": (
            "Algorithm 5 row ownership with separately named tree laws; "
            "graph composition is r0 extension"
        ),
        "conditional_moments": (
            "conditional on retained finite row-owned leaves; downstream laws remain explicit"
        ),
        "target_origin": task.target_origin,
        "runtime_emissions": runtime_card.get(
            "native_emissions", runtime_card.get("converter_emissions")
        ),
        "runtime_diagnostics": runtime_card.get("node_diagnostics"),
        "active_features": runtime_card.get("active"),
        "active_definition": runtime_card.get("active_definition"),
        "moment_contract": runtime_card.get("moment_contract"),
        "flags": runtime_card.get("flags"),
        "intervention": intervention,
        "admission": recipe.request.admit(),
    }
    source = record_fingerprint((config, recipe.seed, recipe.task_id))
    return replace(
        task,
        source_task_id=source,
        episode_id=record_fingerprint((source, recipe.task.split, recipe.task.context_rows)),
        recipe_json=encode(record),
        mechanism_json=encode(card),
    )


def generate_recursive_population_graph(
    recipe: RecursivePopulationGraph, *, seed: int, task_id: int, device: str = "cpu"
) -> TrainingTask:
    population_identity(seed, task_id)
    if (
        type(recipe) is not RecursivePopulationGraph
        or device != "cpu"
        or (seed, task_id) != (recipe.seed, recipe.task_id)
    ):
        raise ValueError("population DAG replay requires its retained identity and CPU execution")
    return population_graph_task(recipe, recipe.replay())
