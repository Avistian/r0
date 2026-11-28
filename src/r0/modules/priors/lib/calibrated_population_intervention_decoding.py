"""Admit baseline calibration and changed population together before any RNG restoration."""

from dataclasses import asdict
from typing import Any

from r0.modules.priors.lib.calibrated_population_intervention import (
    CalibratedPopulationIntervention,
)
from r0.modules.priors.lib.calibrated_population_query import CalibratedPopulationQuery
from r0.modules.priors.lib.calibrated_query_decoding import (
    calibrated_source_from_dict,
    query_source_metadata,
)
from r0.modules.priors.lib.causal_intervention_records import CausalInterventionRecipe
from r0.modules.priors.lib.causal_limits import CausalLimits
from r0.modules.priors.lib.function_metadata import admit_function_metadata
from r0.modules.priors.lib.joint_function_admission import function_record_cost
from r0.modules.priors.lib.population_calibration_custody import preflight_calibration_custody
from r0.modules.priors.lib.population_graph_intervention_decoding import (
    population_graph_intervention_from_dict,
    preflight_population_graph_intervention,
)
from r0.modules.priors.lib.population_graph_semantic_preflight import (
    authored_semantic_preview,
    complete_version,
)
from r0.modules.priors.lib.population_target_calibration import CalibratedPopulationGraph
from r0.modules.priors.lib.recursive_population_decoding import population_raw_cost
from r0.modules.priors.lib.recursive_population_graph_decoding import (
    recursive_population_graph_request_from_dict,
)
from r0.modules.priors.lib.recursive_task_preflight import recursive_authored_preview
from r0.modules.priors.lib.representatives import encode


def _aggregate(
    reports: list[dict[str, int]], numbers: int, storage: int, cap: CausalLimits
) -> None:
    if (
        4 * sum(r["parameters"] for r in reports) + numbers > cap.parameters
        or 4 * sum(r["estimated_working_bytes"] for r in reports) + 8 * storage > cap.working_bytes
    ):
        raise ValueError("aggregate calibrated intervention exceeds caller limits")


def admit_calibrated_change(
    base: CalibratedPopulationGraph | CalibratedPopulationQuery, change: CausalInterventionRecipe
) -> None:
    numbers, storage = function_record_cost((base, change), base.limits)
    _, _, _, _, _, reports, _, _ = query_source_metadata(
        asdict(base), limits=base.limits, verify_streams=False
    )
    _aggregate(
        [*reports, base.original.request.admit(limits=base.limits), change.admit()],
        numbers,
        storage,
        base.limits,
    )


def preflight_calibrated_intervention(
    data: Any, *, limits: CausalLimits, verify_streams: bool = True
):
    admit_function_metadata(data, limits)
    raw = complete_version(data, CalibratedPopulationIntervention)
    numbers, storage = population_raw_cost(raw, limits)
    if numbers >= limits.parameters or 8 * storage >= limits.working_bytes:
        raise ValueError("calibrated intervention metadata exceeds caller limits")
    _, _, current, _, cap, reports, _, _ = query_source_metadata(
        raw["base"], limits=limits, verify_streams=False
    )
    if encode(raw["original"].get("base")) != encode(current):
        raise ValueError("calibrated intervention must retain its exact original population")
    preflight_calibration_custody(raw["original"], cap)
    _, _, _, _, after_report = preflight_population_graph_intervention(raw["original"], limits=cap)
    preview = authored_semantic_preview(recursive_authored_preview(current["request"]))
    request = recursive_population_graph_request_from_dict(preview, limits=cap)
    _aggregate([*reports, request.admit(limits=cap), after_report], numbers, storage, cap)
    if verify_streams:
        query_source_metadata(raw["base"], limits=cap)
    return raw, cap


def calibrated_intervention_from_dict(data: Any, *, limits: CausalLimits):
    try:
        raw, cap = preflight_calibrated_intervention(data, limits=limits)
        result = CalibratedPopulationIntervention(
            calibrated_source_from_dict(raw["base"], limits=cap),
            population_graph_intervention_from_dict(raw["original"], limits=cap),
            raw["version"],
        )
        result.validate(limits=cap)
        if encode(asdict(result)) != encode(data):
            raise ValueError("calibrated intervention differs from canonical archive")
        return result
    except (KeyError, TypeError, AttributeError, IndexError, OverflowError, RecursionError) as exc:
        raise ValueError("invalid complete calibrated population intervention archive") from exc
