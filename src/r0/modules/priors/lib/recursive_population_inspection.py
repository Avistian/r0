"""Inspect full original populations before selecting repeated or out-of-order rows."""

import json
from collections.abc import Callable, Sequence
from dataclasses import asdict
from typing import Any

import numpy as np
import torch

from r0.modules.priors.lib.causal_limits import CausalLimits
from r0.modules.priors.lib.functional_causal_node import FunctionalCausalNode
from r0.modules.priors.lib.recursive_forest_evaluation import (
    evaluate_recursive_forest,
    inspect_recursive_forest,
)
from r0.modules.priors.lib.recursive_population_ancestors import population_ancestor_values
from r0.modules.priors.lib.recursive_population_intervention import RecursivePopulationIntervention
from r0.modules.priors.lib.recursive_population_records import RecursivePopulationRecipe
from r0.modules.priors.lib.training_task import TrainingTask


def recursive_population_steps(
    task: TrainingTask,
    indices: Sequence[int],
    add: Callable[..., None],
    *,
    limits: CausalLimits | None,
) -> dict[str, Any]:
    from r0.modules.priors.lib.generate import generate_task, recipe_from_dict

    record = json.loads(task.recipe_json)
    recipe = recipe_from_dict(record["recipe"], limits=limits or CausalLimits())
    if not isinstance(recipe, (RecursivePopulationRecipe, RecursivePopulationIntervention)):
        raise ValueError("population inspection requires its distinct lifecycle record")
    replay = generate_task(recipe, seed=record["seed"], task_id=record["task_id"])
    expected_record = json.loads(replay.recipe_json)
    if (
        record["execution"].keys() != expected_record["execution"].keys()
        or record["execution"]["implementation"] != expected_record["execution"]["implementation"]
        or any(
            getattr(task, name) != getattr(replay, name)
            for name in (
                "source_task_id",
                "episode_id",
                "task_type",
                "classes",
                "feature_schema",
                "target_origin",
                "mechanism_json",
            )
        )
        or task.probabilities is not None
        or any(
            not torch.equal(getattr(task, name), getattr(replay, name))
            for name in ("inputs", "targets", "scores", "context_indices", "query_indices")
        )
    ):
        raise ValueError("population task differs from exact semantic replay")
    intervention = recipe if isinstance(recipe, RecursivePopulationIntervention) else None
    base = recipe.base if isinstance(recipe, RecursivePopulationIntervention) else recipe
    values = (
        intervention.values()
        if intervention
        else population_ancestor_values(
            base.request.ancestors, seed=base.seed, task_id=base.task_id
        )
    )
    original = list(indices)
    for i, state in enumerate(values.states):
        node = base.request.ancestors.nodes[i]
        equation = node if isinstance(node, FunctionalCausalNode) else node.equation
        add(
            f"population-state-{i}",
            f"Ancestor {i} encoded state",
            [f"population-state-{p}" for p in equation.parents],
            "replay original ancestor equation and emission; descendants receive encoded state",
            state[original],
            {"original_rows": original, "node": asdict(node)},
        )
    parent_rows = recipe.parent_rows
    sample = recipe.sample
    matrix = np.asarray(parent_rows, dtype=np.float64)
    add(
        "population-inputs",
        "Ordered per-row forest inputs",
        list(dict.fromkeys(f"population-state-{i}" for i, _ in base.request.parents)),
        "select declared encoded state channels, preserving original row ownership",
        torch.tensor(matrix[original], dtype=torch.float64),
        {
            "parent_slices": base.request.parents,
            "original_rows": original,
            "forest_indices": original,
        },
    )
    members: list[str] = []
    bypass = intervention is not None and (
        intervention.forced_category is not None
        or intervention.forced_value is not None
        and base.request.plan.output_width == 1
    )
    if not bypass:
        paths = inspect_recursive_forest(sample, matrix, limits=limits or base.limits)
        for tree in range(base.request.plan.estimators):
            leaves = [sample.forests[i].trees[tree].nodes[paths[i][tree][-1]] for i in original]
            emitted = (
                [[n.label] for n in leaves]
                if base.request.plan.classes
                else [n.value for n in leaves]
            )
            key = f"population-tree-{tree}"
            add(
                key,
                f"Row-owned tree slot {tree}",
                ["population-inputs"],
                "each original row traverses its own tree, feature <= threshold goes left",
                torch.tensor(emitted, dtype=torch.float64),
                {
                    "original_rows": original,
                    "paths": [paths[i][tree] for i in original],
                    "leaf_semantics": "actual-class-label"
                    if base.request.plan.classes
                    else "regression-vector",
                },
            )
            members.append(key)
    if intervention and intervention.forced_value is not None and not bypass:
        raw_scores = evaluate_recursive_forest(sample, matrix, limits=limits or base.limits)
        add(
            "population-before-clamp",
            "Raw mean before explicit target clamp",
            members,
            "coordinatewise mean of row-owned tree leaves",
            torch.tensor(raw_scores[original]),
            {},
        )
        members = ["population-before-clamp"]
    add(
        "population-output",
        "Population forest output",
        members or ["population-inputs"],
        "explicit forced target coordinate"
        if intervention and intervention.forced_value is not None
        else "explicit forced category"
        if intervention and intervention.forced_category is not None
        else (
            "majority vote; exact winner indicator, not probabilities"
            if base.request.plan.classes
            else "coordinatewise mean of row-owned tree leaves"
        ),
        replay.scores[original],
        {
            "lifecycle": "fresh-forest-per-row",
            "tie_break": base.request.plan.tie_break,
            "forced_value": intervention.forced_value if intervention else None,
            "forced_category": intervention.forced_category if intervention else None,
        },
    )
    add(
        "targets",
        "Direct native targets",
        ["population-output"],
        "emit actual vocabulary label or selected regression coordinate",
        replay.targets[original, None],
        {"vocabulary": base.request.vocabulary, "target_channel": base.request.target_channel},
    )
    return {
        "version": "generation-graph-v1",
        "recipe_version": recipe.version,
        "episode_id": task.episode_id,
        "rows": original,
        "component": None,
        "component_count": 1,
        "target_error": 0.0,
    }
