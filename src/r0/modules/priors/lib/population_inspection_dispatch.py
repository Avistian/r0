"""Route complete population task inspection by its retained lifecycle envelope."""

from collections.abc import Callable, Sequence
from typing import Any

from r0.modules.priors.lib.causal_limits import CausalLimits
from r0.modules.priors.lib.population_task_semantics import population_task_record
from r0.modules.priors.lib.training_task import TrainingTask


def population_inspection_steps(
    task: TrainingTask,
    indices: Sequence[int],
    add: Callable[..., None],
    *,
    limits: CausalLimits | None,
) -> dict[str, Any]:
    record = population_task_record(task)
    version = record["recipe"].get("version")
    if version in ("calibrated-population-graph-v1", "calibrated-population-intervention-v1"):
        from r0.modules.priors.lib.population_calibration_inspection import (
            calibrated_population_steps,
        )

        return calibrated_population_steps(task, indices, add, limits=limits)
    if version in ("recursive-population-graph-v1", "recursive-population-graph-intervention-v1"):
        from r0.modules.priors.lib.population_graph_inspection import population_graph_steps

        return population_graph_steps(task, indices, add, limits=limits)
    from r0.modules.priors.lib.recursive_population_inspection import recursive_population_steps

    return recursive_population_steps(task, indices, add, limits=limits)
