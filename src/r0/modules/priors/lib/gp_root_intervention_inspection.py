"""Validate complete intervention tasks and inspect bound GP roots and surgical stages."""

import json
from collections.abc import Callable, Sequence
from dataclasses import asdict, replace
from typing import Any

import torch

from r0.modules.priors.lib.causal_intervention_generation import generate_intervened
from r0.modules.priors.lib.causal_intervention_inspection import intervention_steps
from r0.modules.priors.lib.causal_limits import CausalLimits
from r0.modules.priors.lib.gp_column_inspection import gp_population_matrices
from r0.modules.priors.lib.gp_root_dags import GPRootDAGRecipe
from r0.modules.priors.lib.gp_root_intervention_decoding import (
    gp_root_intervention_from_dict,
    preflight_gp_intervention,
)
from r0.modules.priors.lib.gp_root_intervention_output import generate_gp_root_intervention
from r0.modules.priors.lib.gp_root_task_validation import gp_root_task_record
from r0.modules.priors.lib.gp_task_validation import validate_gp_execution, validate_gp_task
from r0.modules.priors.lib.representatives import record_fingerprint
from r0.modules.priors.lib.training_task import TrainingTask


def validate_gp_intervention_task(task: TrainingTask, *, limits: CausalLimits):
    record = gp_root_task_record(task)
    raw, metadata = preflight_gp_intervention(record["recipe"], limits=limits)
    original = metadata.current["source"]
    if any(type(record[k]) is not int or record[k] != original[k] for k in ("seed", "task_id")):
        raise ValueError("GP intervention task requires exact original identity")
    validate_gp_execution(record["execution"], raw["version"])
    if record["execution"]["device"] != "cpu":
        raise ValueError("GP intervention task requires CPU execution")
    source = record_fingerprint((raw, record["seed"], record["task_id"]))
    controls = metadata.base.task
    if (
        task.source_task_id != source
        or task.episode_id != record_fingerprint((source, controls.split, controls.context_rows))
        or task.task_type != controls.task_type
    ):
        raise ValueError("GP intervention source, episode or task semantics differ")
    recipe = gp_root_intervention_from_dict(raw, limits=limits)
    expected = generate_gp_root_intervention(recipe, seed=record["seed"], task_id=record["task_id"])
    validate_gp_task(task, expected, recipe.version)
    return recipe, expected


def gp_intervention_steps(
    task: TrainingTask,
    indices: Sequence[int],
    add: Callable[..., None],
    *,
    limits: CausalLimits | None,
) -> dict[str, Any]:
    cap = limits or CausalLimits()
    recipe, expected = validate_gp_intervention_task(task, limits=cap)
    record = json.loads(task.recipe_json)
    context = replace(recipe.original, source=recipe.population).context()
    rows = list(indices)
    add(
        "gp-population",
        "Retained intervention GP population",
        [],
        "joint draw at all retained population rows; never a subset refit",
        torch.tensor(recipe.population.draw.values, dtype=torch.float64)[rows],
        {"population_id": recipe.population.population_id, "mode": recipe.change.mode},
    )
    for binding in recipe.original.bindings:
        values = context.values(binding.node, recipe.task.rows, len(binding.columns), "cpu")
        assert values is not None
        add(
            f"gp-binding-{binding.node}",
            f"Retained root input {binding.node}",
            ["gp-population"],
            "select declared original GP columns; a surgical bypass may leave these unused",
            values[rows],
            asdict(binding),
        )
    calibrated = not isinstance(recipe.base, GPRootDAGRecipe)
    binding_nodes = {b.node for b in recipe.original.bindings}

    def bound_add(
        key: str,
        label: str,
        parents: list[str],
        formula: str,
        values: torch.Tensor,
        parameters: dict[str, Any],
    ) -> None:
        parts = key.split("-")
        if (
            len(parts) == 3
            and parts[0] == "intervention"
            and parts[1].isdigit()
            and int(parts[1]) in binding_nodes
            and parts[2] in ("root", "inputs")
            and not parents
        ):
            parents = [f"gp-binding-{parts[1]}"]
        if key == "targets" and calibrated:
            key, label = "raw-targets", "Uncalibrated intervention target"
        add(key, label, parents, formula, values, parameters)

    raw_task = generate_intervened(
        recipe.change, seed=record["seed"], task_id=record["task_id"], root_populations=context
    )
    trace = intervention_steps(raw_task, indices, bound_add, limits=cap, root_populations=context)
    if not isinstance(recipe.base, GPRootDAGRecipe):
        fit = recipe.base.calibration
        add(
            "raw-scores",
            "Raw structural intervention scores",
            [],
            "declared structural target readout",
            raw_task.scores[rows],
            {},
        )
        add(
            "fit-statistics",
            "Fixed independent target statistics",
            [],
            "all independent fit rows; constant channels retain identity center 0 and scale 1",
            torch.tensor((fit.center, fit.scale), dtype=torch.float64),
            {
                **asdict(fit),
                "row_labels": ["center", "scale"],
                "row_heading": "Statistic",
                "column_heading": "Target score channel",
                "axes": f"Center and scale over all {fit.rows} independent fitting rows.",
                "scope": "global fitting statistics, not an emitted dataset row",
                "population_id": recipe.base.fit.source.population_id,
                "seed": recipe.base.fit.source.seed,
                "task_id": recipe.base.fit.source.task_id,
                "query_fit": False,
            },
        )
        add(
            "scores",
            "Calibrated intervention scores",
            ["raw-scores", "fit-statistics"],
            "(raw score - fixed center) / fixed scale; no query fitting",
            expected.scores[rows],
            {},
        )
        add(
            "targets",
            "Intervention targets with retained calibration",
            ["scores"],
            "fixed calibration then original target adapter or direct numerical target",
            expected.targets[rows, None],
            {},
        )
        if expected.probabilities is not None:
            add(
                "probabilities",
                "Intervention class probabilities",
                ["scores"],
                "retained target adapter",
                expected.probabilities[rows],
                {},
            )
    return {
        **trace,
        "kind": "gp_root_intervention",
        "episode_id": task.episode_id,
        "population_id": recipe.population.population_id,
        "original_population_id": recipe.original.source.population_id,
        "shared_matrices": gp_population_matrices(recipe.population),
        "bindings": [asdict(b) for b in recipe.original.bindings],
    }
