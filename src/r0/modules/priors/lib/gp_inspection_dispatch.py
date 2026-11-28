"""Dispatch GP finite populations, target environments and hybrid root traces."""

from collections.abc import Callable, Sequence
from typing import Any

from r0.modules.priors.lib.causal_limits import CausalLimits
from r0.modules.priors.lib.training_task import TrainingTask


def gp_inspection(
    task: TrainingTask,
    raw: dict[str, Any],
    indices: Sequence[int],
    add: Callable[..., None],
    *,
    component: int | None,
    limits: CausalLimits | None,
) -> dict[str, Any] | None:
    if raw.get("version") == "gp-root-calibration-v1":
        from r0.modules.priors.lib.gp_root_calibration_inspection import calibrated_gp_root_steps

        if component is not None:
            raise ValueError("GP calibrated DAG has no composite component")
        return calibrated_gp_root_steps(task, indices, add, limits=limits)

    if raw.get("version") == "gp-root-dag-v1":
        from r0.modules.priors.lib.gp_root_inspection import gp_root_steps

        if component is not None:
            raise ValueError("GP root DAG has no composite component")
        return {**gp_root_steps(task, indices, add, limits=limits)}

    if raw.get("version") in ("gp-column-target-recipe-v1", "gp-column-target-link-v1"):
        from r0.modules.priors.lib.gp_target_inspection import gp_target_steps

        if component is not None:
            raise ValueError("GP target environments do not have composite children")
        return {**gp_target_steps(task, indices, add, limits=limits)}

    if raw.get("version") == "gp-column-recipe-v1":
        from r0.modules.priors.lib.gp_column_inspection import gp_column_steps

        if component is not None:
            raise ValueError("GP column populations do not have composite children")
        return {**gp_column_steps(task, indices, add, limits=limits)}

    return None
