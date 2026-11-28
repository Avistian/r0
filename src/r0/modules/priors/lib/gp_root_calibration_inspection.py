"""Original-row GP DAG traces followed by retained independent target scaling."""

from collections.abc import Callable, Sequence
from dataclasses import asdict
from typing import Any

import torch

from r0.modules.priors.lib.causal_limits import CausalLimits
from r0.modules.priors.lib.functional_causal import FunctionalCausalRecipe
from r0.modules.priors.lib.functional_causal_evaluation import functional_rng
from r0.modules.priors.lib.gp_column_inspection import gp_population_matrices
from r0.modules.priors.lib.gp_root_calibration import GPRootCalibration
from r0.modules.priors.lib.gp_root_dags import generate_gp_root_dag
from r0.modules.priors.lib.gp_root_inspection import gp_root_steps
from r0.modules.priors.lib.gp_root_task_validation import validate_gp_root_task
from r0.modules.priors.lib.target_inspection import target_steps
from r0.modules.priors.lib.training_task import TrainingTask


def calibrated_gp_root_steps(
    task: TrainingTask,
    indices: Sequence[int],
    add: Callable[..., None],
    *,
    limits: CausalLimits | None,
) -> dict[str, Any]:
    recipe, expected = validate_gp_root_task(task, limits=limits)
    assert isinstance(recipe, GPRootCalibration)
    source = recipe.original.source
    original = generate_gp_root_dag(recipe.original, seed=source.seed, task_id=source.task_id)
    ids: set[str] = set()

    def original_add(
        key: str,
        label: str,
        parents: list[str],
        formula: str,
        values: torch.Tensor,
        parameters: dict[str, Any],
    ):
        ids.add(key)
        add(
            "original-" + key,
            label,
            ["original-" + p for p in parents],
            formula,
            values,
            parameters,
        )

    gp_root_steps(original, indices, original_add, limits=limits)
    parent = next((key for key in ("signal", "scores", "targets", "target") if key in ids), None)
    rows = list(indices)
    add(
        "raw-signal",
        "Original GP DAG target readout",
        [] if parent is None else ["original-" + parent],
        "complete original DAG readout before fixed target calibration",
        original.scores[rows],
        {"row_ids": rows},
    )
    add(
        "fit-statistics",
        "Retained independent GP calibration statistics",
        [],
        "center and scale fitted on all independent GP DAG rows; constants keep identity scaling",
        torch.tensor([recipe.calibration.center, recipe.calibration.scale], dtype=torch.float64),
        {
            **asdict(recipe.calibration),
            "row_labels": ["center", "scale"],
            "row_heading": "Statistic",
            "column_heading": "Target score channel",
            "axes": (
                f"Center and scale over all {recipe.calibration.rows} independent fitting rows."
            ),
            "scope": "global fitting statistics, not an emitted dataset row",
            "population_id": recipe.fit.source.population_id,
            "seed": recipe.fit.source.seed,
            "task_id": recipe.fit.source.task_id,
            "query_fit": False,
        },
    )
    signal = recipe.calibration.apply(original.scores)[rows]
    add(
        "signal",
        "Calibrated target readout",
        ["raw-signal", "fit-statistics"],
        "(raw_signal - retained_center) / retained_scale",
        signal,
        {"query_fit": False, "fit_population_id": recipe.fit.source.population_id},
    )
    if isinstance(recipe.original.base, FunctionalCausalRecipe):
        targets = target_steps(
            asdict(recipe.task),
            signal,
            indices,
            lambda key: functional_rng(source.seed, source.task_id, key),
            add,
        )
    else:
        targets = signal[:, 0]
        add("target", "Calibrated numeric target", ["signal"], "signal[:, 0]", targets, {})
    if not torch.allclose(targets, expected.targets[rows], atol=1e-10, rtol=1e-10):
        raise ValueError("GP calibrated target trace differs from retained task")
    return {
        "version": "generation-graph-v1",
        "kind": "gp_root_calibration",
        "episode_id": task.episode_id,
        "rows": rows,
        "population_id": source.population_id,
        "shared_matrices": gp_population_matrices(source),
        "fit_population_id": recipe.fit.source.population_id,
        "query_fit": False,
        "component": None,
        "component_count": 0,
    }
