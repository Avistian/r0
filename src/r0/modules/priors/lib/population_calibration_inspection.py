"""Independent calibration population and fixed target transform in original-row traces."""

from collections.abc import Callable, Sequence
from dataclasses import asdict
from typing import Any

import numpy as np
import torch

from r0.modules.priors.lib.causal_intervention_execution import InterventionExecution
from r0.modules.priors.lib.causal_limits import CausalLimits
from r0.modules.priors.lib.functional_causal import FunctionalCausalRecipe
from r0.modules.priors.lib.functional_causal_evaluation import functional_rng
from r0.modules.priors.lib.population_calibrated_output import generate_calibrated_population
from r0.modules.priors.lib.population_calibration_decoding import calibrated_population_from_dict
from r0.modules.priors.lib.population_calibration_inspection_identity import (
    preflight_calibrated_inspection,
)
from r0.modules.priors.lib.population_graph_inspection import population_graph_steps
from r0.modules.priors.lib.population_task_semantics import (
    population_task_record,
    validate_population_task,
)
from r0.modules.priors.lib.recursive_population_graph_output import (
    generate_recursive_population_graph,
)
from r0.modules.priors.lib.target_inspection import target_steps
from r0.modules.priors.lib.training_task import TrainingTask


def calibrated_population_steps(
    task: TrainingTask,
    indices: Sequence[int],
    add: Callable[..., None],
    *,
    limits: CausalLimits | None,
) -> dict[str, Any]:
    record = population_task_record(task)
    preflight_calibrated_inspection(task, record)
    cap = limits or CausalLimits()
    execution: InterventionExecution | None = None
    if record["recipe"].get("version") == "calibrated-population-intervention-v1":
        from r0.modules.priors.lib.calibrated_population_intervention_decoding import (
            calibrated_intervention_from_dict,
        )
        from r0.modules.priors.lib.calibrated_population_intervention_output import (
            generate_calibrated_intervention,
        )
        from r0.modules.priors.lib.recursive_population_graph_intervention import (
            generate_population_graph_intervention,
        )

        changed = calibrated_intervention_from_dict(record["recipe"], limits=cap)
        recipe = changed.base
        replay = generate_calibrated_intervention(
            changed, seed=record["seed"], task_id=record["task_id"]
        )
        original = generate_population_graph_intervention(
            changed.original, seed=record["seed"], task_id=record["task_id"]
        )
        execution = InterventionExecution(changed.original.change)
        role = "counterfactual raw population; baseline calibration remains fixed"
    elif record["recipe"].get("version") == "calibrated-population-query-v1":
        from r0.modules.priors.lib.calibrated_query_decoding import calibrated_query_from_dict
        from r0.modules.priors.lib.calibrated_query_output import generate_calibrated_query

        recipe = calibrated_query_from_dict(record["recipe"], limits=cap)
        replay = generate_calibrated_query(recipe, seed=record["seed"], task_id=record["task_id"])
        original = generate_recursive_population_graph(
            recipe.original, seed=record["seed"], task_id=record["task_id"]
        )
        role = "fresh query population; complete source lineage and baseline calibration retained"
    else:
        recipe = calibrated_population_from_dict(record["recipe"], limits=cap)
        replay = generate_calibrated_population(
            recipe, seed=record["seed"], task_id=record["task_id"]
        )
        original = generate_recursive_population_graph(
            recipe.original, seed=recipe.original.seed, task_id=recipe.original.task_id
        )
        role = "retained original population"
    validate_population_task(task, replay, record)

    def rng(key: str) -> np.random.Generator:
        if execution is not None:
            return execution.rng(
                lambda *keys: functional_rng(record["seed"], record["task_id"], *keys), key
            )
        return functional_rng(record["seed"], record["task_id"], key)

    def original_step(
        key: str,
        label: str,
        parents: list[str],
        formula: str,
        values: torch.Tensor,
        parameters: dict[str, Any],
    ) -> None:
        add(
            f"original-{key}",
            label,
            [f"original-{p}" for p in parents],
            formula,
            values,
            {**parameters, "calibration_role": role},
        )

    population_graph_steps(original, indices, original_step, limits=cap)
    fit = recipe.calibration_population
    fit_scores = fit.replay(limits=cap).task.scores
    add(
        "calibration-population",
        "Independent target calibration scores",
        [],
        "same frozen graph laws; separately keyed row-owned forests and exogenous draws",
        fit_scores,
        {
            "scope": "independent calibration population, not selected dataset rows",
            "row_ids": list(range(fit.task.rows)),
            "seed": fit.seed,
            "task_id": fit.task_id,
            "query_fit": False,
            "sample_digest": recipe.calibration.sample_digest,
        },
    )
    rows = list(indices)
    functional = isinstance(recipe.original.request.base, FunctionalCausalRecipe)
    add(
        "signal",
        "Fixed independently calibrated target channels",
        ["original-signal" if functional else "original-target", "calibration-population"],
        "(original target channels - retained independent center) / retained independent scale",
        replay.scores[rows],
        {
            **asdict(recipe.calibration),
            "constant_policy": "identity",
            "original_rows": rows,
            "query_fit": False,
        },
    )
    if functional:
        targets = target_steps(
            asdict(recipe.task),
            replay.scores[rows],
            indices,
            rng,
            add,
        )
        if not torch.equal(targets, replay.targets[rows]):
            raise ValueError("calibrated population target adapter differs from replay")
    else:
        add(
            "target",
            "Calibrated direct continuous target",
            ["signal"],
            "select fixed calibrated continuous channel, without adding a target adapter",
            replay.targets[rows, None],
            {"target_origin": replay.target_origin, "classes": replay.classes},
        )
    return {
        "version": "generation-graph-v1",
        "recipe_version": record["recipe"]["version"],
        "episode_id": task.episode_id,
        "rows": rows,
        "component": None,
        "component_count": 1,
        "signal_max_error": 0.0,
        "target_max_error": 0.0,
    }
