"""Inspect retained calibration rows without fitting or replacing the active query."""

from collections.abc import Sequence
from dataclasses import asdict
from typing import Any

from r0.modules.priors.lib.causal_limits import CausalLimits
from r0.modules.priors.lib.gp_root_calibration import GPRootCalibration
from r0.modules.priors.lib.gp_root_calibration_decoding import preflight_gp_root_calibration
from r0.modules.priors.lib.gp_root_dags import generate_gp_root_dag
from r0.modules.priors.lib.gp_root_query import GPRootCalibrationQuery
from r0.modules.priors.lib.gp_root_query_metadata import query_source_metadata
from r0.modules.priors.lib.gp_root_task_validation import gp_root_task_record, validate_gp_root_task
from r0.modules.priors.lib.training_task import TrainingTask


def explain_gp_calibration_fit(
    task: TrainingTask,
    indices: Sequence[int],
    *,
    component: int | None = None,
    column_start: int = 0,
    limits: CausalLimits | None = None,
) -> dict[str, Any]:
    """Trace independent fitting rows, with full retained population matrices.

    The active query task is validated first. Its fitting graph is then replayed
    under its own identity. Statistics always describe all fitting rows, never
    just the selected trace rows. Neither population nor calibration is modified.
    """
    from r0.modules.priors.lib.generation_graph import explain_generation

    if limits is not None and type(limits) is not CausalLimits:
        raise ValueError("fit inspection requires explicit causal limits")
    if component is not None:
        raise ValueError("GP fit population has no composite component")
    if type(column_start) is not int or not 0 <= column_start <= 4095:
        raise ValueError("fit inspection column_start must be 0..4095")
    if not 1 <= len(indices) <= 16:
        raise ValueError("select 1..16 fit population rows")
    raw = gp_root_task_record(task)["recipe"]
    if raw.get("version") == "gp-root-calibration-query-v1":
        query_source_metadata(raw, limits=limits or CausalLimits())
        while raw["version"] == "gp-root-calibration-query-v1":
            raw = raw["source"]
    if raw.get("version") != "gp-root-calibration-v1":
        raise ValueError("fit inspection requires a retained GP calibration")
    _, calibration = preflight_gp_root_calibration(raw, limits=limits or CausalLimits())
    if any(type(i) is not int or not 0 <= i < calibration.rows for i in indices):
        raise ValueError("invalid fit population row index")
    recipe, _ = validate_gp_root_task(task, limits=limits)
    assert isinstance(recipe, (GPRootCalibration, GPRootCalibrationQuery))
    source = recipe.fit.source
    fitting = generate_gp_root_dag(recipe.fit, seed=source.seed, task_id=source.task_id)
    graph = explain_generation(fitting, indices, column_start=column_start, limits=limits)
    start = min(column_start, len(calibration.center) - 1)
    end = min(start + 32, len(calibration.center))
    graph["nodes"].append(
        dict(
            id="fit-statistics",
            label="Fixed calibration statistics from every fitting row",
            parents=[],
            formula="mean and population std over all independent fit raw scores",
            shape=[2, len(calibration.center)],
            column_indices=list(range(start, end)),
            values=[list(calibration.center[start:end]), list(calibration.scale[start:end])],
            parameters={
                **asdict(calibration),
                "row_labels": ["center", "scale"],
                "row_heading": "Statistic",
                "column_heading": "Target score channel",
                "axes": f"Center and scale over all {calibration.rows} independent fitting rows.",
                "scope": "global fitting statistics, not an emitted dataset row",
                "displayed_fit_rows": list(indices),
                "query_fit": False,
                "population_id": source.population_id,
            },
        )
    )
    return {
        **graph,
        "kind": "gp_root_fit",
        "episode_id": task.episode_id,
        "fit_episode_id": fitting.episode_id,
        "population_role": "fit",
        "fit_seed": source.seed,
        "fit_task_id": source.task_id,
        "total_population_rows": calibration.rows,
        "query_fit": False,
    }
