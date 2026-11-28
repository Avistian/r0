"""Strict shared comparison for retained row-owned DAG population pairs."""

from dataclasses import asdict
from typing import Any

import torch

from r0.modules.priors.lib.causal_limits import CausalLimits
from r0.modules.priors.lib.exact_frozen_record import exact_frozen_equal
from r0.modules.priors.lib.population_task_semantics import (
    population_task_record,
    validate_population_task,
)
from r0.modules.priors.lib.recursive_population_graph_intervention import (
    RecursivePopulationGraphIntervention,
)
from r0.modules.priors.lib.recursive_population_graph_records import RecursivePopulationGraph
from r0.modules.priors.lib.representatives import encode
from r0.modules.priors.lib.training_task import TrainingTask


def population_graph_pair_card(
    base: RecursivePopulationGraph,
    changed: RecursivePopulationGraphIntervention,
    before: TrainingTask,
    after: TrainingTask,
) -> dict[str, Any]:
    return {
        "version": "recursive-population-graph-pair-v1",
        "mode": changed.change.mode,
        "population": changed.change.population,
        "before_episode": before.episode_id,
        "after_episode": after.episode_id,
        "base_archive_unchanged": exact_frozen_equal(base, changed.base),
        "forests_equal": exact_frozen_equal(
            tuple(b.sample for b in base.bindings), changed.samples
        ),
        "node_inputs_equal": [
            {
                "node": b.node,
                "equal": exact_frozen_equal(b.inputs, dict(changed.inputs)[b.node]),
            }
            for b in base.bindings
        ],
        "inputs_equal": bool(torch.equal(before.inputs, after.inputs)),
        "targets_equal": bool(torch.equal(before.targets, after.targets)),
        "split_equal": bool(torch.equal(before.context_indices, after.context_indices)),
        "variance_semantics": (
            "raw row-owned mechanisms; no variance matching or query calibration"
        ),
    }


def compare_population_graph_pair(
    before: TrainingTask, after: TrainingTask, *, limits: CausalLimits | None
) -> dict[str, Any]:
    from r0.modules.priors.lib.generate import generate_task, recipe_from_dict

    b, a = population_task_record(before), population_task_record(after)
    raw = a["recipe"]
    if (
        raw.get("version") != "recursive-population-graph-intervention-v1"
        or type(raw.get("base")) is not dict
    ):
        raise ValueError("population comparison requires its complete intervention archive")
    if encode(b["recipe"]) != encode(raw["base"]):
        raise ValueError("population comparison must retain its exact original base archive")
    if (b["seed"], b["task_id"]) != (a["seed"], a["task_id"]):
        raise ValueError("population comparison requires original source row identities")
    if any(
        record["execution"]["implementation"] != record["recipe"].get("version")
        for record in (b, a)
    ):
        raise ValueError("population comparison implementation identity differs before replay")
    # The intervention decoder preadmits both populations and complete metadata before restore.
    recipe = recipe_from_dict(raw, limits=limits)
    if not isinstance(recipe, RecursivePopulationGraphIntervention):
        raise ValueError("population comparison requires a row-owned DAG intervention")
    seed, task_id = a["seed"], a["task_id"]
    expected_before = generate_task(recipe.base, seed=seed, task_id=task_id)
    expected_after = generate_task(recipe, seed=seed, task_id=task_id)
    validate_population_task(before, expected_before, b)
    validate_population_task(after, expected_after, a)
    if encode(asdict(recipe.base)) != encode(b["recipe"]):
        raise ValueError("population comparison original construction differs")
    return population_graph_pair_card(recipe.base, recipe, before, after)
