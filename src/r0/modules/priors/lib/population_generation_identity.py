"""Cheap frozen population identity checks precede calibration or source replay."""

from typing import Any, cast

from r0.modules.priors.lib.population_target_calibration import CalibratedPopulationGraph
from r0.modules.priors.lib.recursive_population_graph_records import RecursivePopulationGraph
from r0.modules.priors.lib.recursive_population_records import population_identity


def preflight_calibrated_generation(
    source: object, *, seed: int, task_id: int, device: str = "cpu", count: int = 1
) -> None:
    """Admit typed or serialized fixed-population transport without restoring any stream."""
    population_identity(seed, task_id)
    if type(device) is not str or device != "cpu":
        raise ValueError("calibrated population generation requires CPU execution")
    if type(count) is not int or count != 1:
        raise ValueError("calibrated population generation requires count=1 for its fixed identity")
    source = _baseline_source(source)
    if type(source) is CalibratedPopulationGraph:
        if (
            source.version != "calibrated-population-graph-v1"
            or type(source.original) is not RecursivePopulationGraph
        ):
            raise ValueError("calibrated generation requires its retained original identity")
        expected = (source.original.seed, source.original.task_id)
    elif type(source) is dict:
        raw = cast(dict[str, Any], source)
        original = raw.get("original")
        if raw.get("version") != "calibrated-population-graph-v1" or type(original) is not dict:
            raise ValueError("calibrated generation requires its retained original identity")
        original = cast(dict[str, Any], original)
        expected = (original.get("seed"), original.get("task_id"))
    else:
        raise ValueError("calibrated generation requires its retained original identity")
    population_identity(cast(int, expected[0]), cast(int, expected[1]))
    if (seed, task_id) != expected:
        raise ValueError("calibrated population generation requires its exact retained identity")


def _baseline_source(source: object) -> object:
    from r0.modules.priors.lib.calibrated_population_intervention import (
        CalibratedPopulationIntervention,
    )
    from r0.modules.priors.lib.recursive_population_graph_intervention import (
        RecursivePopulationGraphIntervention,
    )

    if type(source) is CalibratedPopulationIntervention:
        if (
            source.version != "calibrated-population-intervention-v1"
            or type(source.original) is not RecursivePopulationGraphIntervention
        ):
            raise ValueError("calibrated intervention requires its retained original identity")
        return source.base
    elif (
        type(source) is dict
        and cast(dict[str, Any], source).get("version") == "calibrated-population-intervention-v1"
    ):
        return cast(dict[str, Any], source).get("base")
    return cast(object, source)
