"""Admit a prospective retained query envelope before replay or new random work."""

from dataclasses import asdict
from typing import Any, cast

from r0.modules.priors.lib.calibrated_population_query import ENGINES, CalibratedPopulationQuery
from r0.modules.priors.lib.calibrated_query_decoding import (
    admit_query_aggregate,
    query_source_metadata,
)
from r0.modules.priors.lib.causal_limits import CausalLimits
from r0.modules.priors.lib.joint_function_admission import function_record_cost
from r0.modules.priors.lib.population_calibration_decoding import calibration_population_metadata
from r0.modules.priors.lib.population_target_calibration import CalibratedPopulationGraph
from r0.modules.priors.lib.recursive_population_decoding import population_raw_cost
from r0.modules.priors.lib.recursive_population_records import population_identity


def preflight_calibrated_query(
    source: object,
    *,
    seed: int,
    task_id: int,
    engine: str | None = None,
    construction_seed: int | None = None,
    limits: CausalLimits | None = None,
) -> CausalLimits:
    """Check typed/raw source and complete future custody without restoring or drawing.

    Raw callers supply their own limits (defaults are conservative). The returned
    effective cap can be reused for source decoding. A discarded same-law shape
    preview adds the new wrapper and population; it is never a realized receipt.
    Worst-case request reservations bound variable forest size and RNG state size.
    """
    population_identity(seed, task_id)
    population_identity(seed if construction_seed is None else construction_seed, task_id)
    if limits is not None and type(limits) is not CausalLimits:
        raise ValueError("calibrated query requires explicit caller limits")
    if engine is not None and (type(engine) is not str or engine not in ENGINES):
        raise ValueError("calibrated query requires an explicit supported NumPy engine")
    if type(source) in (CalibratedPopulationGraph, CalibratedPopulationQuery):
        typed = cast(CalibratedPopulationGraph | CalibratedPopulationQuery, source)
        cap = limits or typed.limits
        function_record_cost(typed, cap)
        raw: Any = asdict(typed)
    elif type(source) is dict:
        cap = limits or CausalLimits()
        raw = cast(dict[str, Any], source)
    else:
        raise ValueError("fresh calibrated query requires a complete retained source")
    _, baseline, current, _, cap, reports, _, _ = query_source_metadata(
        raw, limits=cap, verify_streams=False
    )
    identities = {
        (baseline["original"]["seed"], baseline["original"]["task_id"]),
        (baseline["calibration_population"]["seed"], baseline["calibration_population"]["task_id"]),
    }
    ancestor = raw
    while ancestor.get("version") == "calibrated-population-query-v1":
        identities.add((ancestor["original"]["seed"], ancestor["original"]["task_id"]))
        ancestor = ancestor["source"]
    if (seed, task_id) in identities:
        raise ValueError(
            "fresh query identity must differ from every source and calibration population"
        )
    selected = (
        engine if engine is not None else current["bindings"][0]["sample"]["initial"]["engine"]
    )
    if type(selected) is not str or selected not in ENGINES:
        raise ValueError("calibrated query requires an explicit supported NumPy engine")
    future = dict(
        source=raw,
        original={**current, "seed": seed, "task_id": task_id},
        engine=selected,
        construction_seed=seed if construction_seed is None else construction_seed,
        version="calibrated-population-query-v1",
    )
    # Full envelope depth and metadata traversal limits apply before source replay.
    numbers, storage = population_raw_cost(future, cap)
    old_numbers, old_storage = population_raw_cost(current, cap)
    _, request = calibration_population_metadata(current, cap)
    report = request.admit(limits=cap)
    # Replace the shape preview's measured new-population metadata charge with
    # at least the existing worst-case request reservation. This includes all
    # forest outcomes, retained inputs and bounded snapshots for any engine.
    numbers += max(0, report["parameters"] - old_numbers)
    storage += max(0, (report["estimated_working_bytes"] + 7) // 8 - old_storage)
    admit_query_aggregate([*reports, report], numbers, storage, cap)
    return cap
