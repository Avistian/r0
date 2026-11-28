"""Whole fixed-fit source custody plus reference workspace before any nested replay."""

from dataclasses import replace
from typing import Any

from r0.modules.priors.lib.calibrated_population_intervention_decoding import (
    preflight_calibrated_intervention,
)
from r0.modules.priors.lib.calibrated_query_decoding import (
    admit_query_aggregate,
    query_source_metadata,
)
from r0.modules.priors.lib.causal_limits import CausalLimits
from r0.modules.priors.lib.population_graph_intervention_decoding import (
    preflight_population_graph_intervention,
)
from r0.modules.priors.lib.recursive_population_decoding import population_raw_cost

CALIBRATED_REFERENCE_SOURCES = (
    "calibrated-population-graph-v1",
    "calibrated-population-query-v1",
    "calibrated-population-intervention-v1",
)


def calibrated_reference_reservation(source: dict[str, Any], limits: CausalLimits):
    if source.get("version") == "calibrated-population-intervention-v1":
        _, cap = preflight_calibrated_intervention(source, limits=limits, verify_streams=False)
        _, _, _, _, _, reports, _, _ = query_source_metadata(
            source["base"], limits=cap, verify_streams=False
        )
        *_, after_report = preflight_population_graph_intervention(source["original"], limits=cap)
        reports = [*reports, after_report]
    else:
        _, _, _, _, cap, reports, _, _ = query_source_metadata(
            source, limits=limits, verify_streams=False
        )
    numbers, storage = population_raw_cost(source, cap)
    admit_query_aggregate(reports, numbers, storage, cap)
    parameters = 4 * sum(r["parameters"] for r in reports) + numbers
    working = 4 * sum(r["estimated_working_bytes"] for r in reports) + 8 * storage
    if parameters >= cap.parameters or working >= cap.working_bytes:
        raise ValueError("calibrated reference needs retained source and query workspace")
    return source["original"], replace(
        cap, parameters=cap.parameters - parameters, working_bytes=cap.working_bytes - working
    )
