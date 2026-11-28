"""Admit the full retained query lineage before source or current population restoration."""

from dataclasses import asdict
from typing import Any, cast

from r0.modules.priors.lib.calibrated_population_query import (
    ENGINES,
    CalibratedPopulationQuery,
    query_stream,
)
from r0.modules.priors.lib.causal_limits import CausalLimits
from r0.modules.priors.lib.function_metadata import admit_function_metadata
from r0.modules.priors.lib.generator_snapshot import snapshot_generator
from r0.modules.priors.lib.population_calibration_custody import preflight_calibration_custody
from r0.modules.priors.lib.population_calibration_decoding import (
    calibrated_population_from_dict,
    calibration_population_metadata,
    preflight_calibrated_population,
)
from r0.modules.priors.lib.population_calibration_identity import validate_calibration_identity
from r0.modules.priors.lib.population_graph_semantic_preflight import complete_version
from r0.modules.priors.lib.population_target_calibration import CalibratedPopulationGraph
from r0.modules.priors.lib.recursive_population_decoding import population_raw_cost
from r0.modules.priors.lib.recursive_population_graph_decoding import (
    recursive_population_graph_from_dict,
)
from r0.modules.priors.lib.recursive_population_records import population_identity
from r0.modules.priors.lib.representatives import encode


def admit_query_aggregate(
    reports: list[dict[str, int]], numbers: int, storage: int, cap: CausalLimits
) -> None:
    if (
        4 * sum(r["parameters"] for r in reports) + numbers > cap.parameters
        or 4 * sum(r["estimated_working_bytes"] for r in reports) + 8 * storage > cap.working_bytes
    ):
        raise ValueError("complete calibrated query lineage exceeds caller limits")


def query_source_metadata(data: Any, *, limits: CausalLimits, verify_streams: bool = True):
    admit_function_metadata(data, limits)
    numbers, storage = population_raw_cost(data, limits)
    if numbers >= limits.parameters or 8 * storage >= limits.working_bytes:
        raise ValueError("calibrated query lineage metadata exceeds caller limits")
    current = data
    lineage: list[dict[str, Any]] = []
    while (
        type(current) is dict
        and cast(dict[str, Any], current).get("version") == "calibrated-population-query-v1"
    ):
        if len(lineage) >= 64:
            raise ValueError("calibrated query lineage exceeds its bounded depth")
        raw = complete_version(current, CalibratedPopulationQuery)
        if type(raw["engine"]) is not str or raw["engine"] not in ENGINES:
            raise ValueError("calibrated query engine differs")
        population_identity(raw["construction_seed"], 0)
        lineage.append(raw)
        current = raw["source"]
    base, calibration, cap = preflight_calibrated_population(
        current, limits=limits, validate_identity=False
    )
    preflight_calibration_custody(data, cap)
    _, original_request = calibration_population_metadata(base["original"], cap)
    _, fit_request = calibration_population_metadata(base["calibration_population"], cap)
    reports = [r.admit(limits=cap) for r in (original_request, fit_request)]
    previous = base["original"]
    identities = {
        (previous["seed"], previous["task_id"]),
        (base["calibration_population"]["seed"], base["calibration_population"]["task_id"]),
    }
    for item in reversed(lineage):
        original, request = calibration_population_metadata(item["original"], cap)
        if encode(original["request"]) != encode(previous["request"]):
            raise ValueError("query geometry/law changes require explicit new calibration")
        identity = original["seed"], original["task_id"]
        if identity in identities:
            raise ValueError("fresh query identity duplicates source lineage or calibration")
        identities.add(identity)
        if any(
            binding["sample"]["initial"]["engine"] != item["engine"]
            for binding in original["bindings"]
        ):
            raise ValueError("calibrated query engine differs from construction receipts")
        reports.append(request.admit(limits=cap))
        previous = original
    admit_query_aggregate(reports, numbers, storage, cap)
    if verify_streams:
        validate_calibration_identity(base["calibration_population"])
        for item in lineage:
            original = item["original"]
            rng = query_stream(
                original["request"],
                original["seed"],
                original["task_id"],
                item["engine"],
                item["construction_seed"],
            )
            expected = snapshot_generator(
                rng, byte_limit=original["request"]["raw_limits"]["state_bytes"]
            )
            if encode(asdict(expected)) != encode(original["bindings"][0]["sample"]["initial"]):
                raise ValueError(
                    "calibrated query construction differs from canonical independent identity"
                )
    return data, base, previous, calibration, cap, reports, numbers, storage


def calibrated_query_from_dict(data: Any, *, limits: CausalLimits) -> CalibratedPopulationQuery:
    try:
        raw, _, _, _, cap, _, _, _ = query_source_metadata(data, limits=limits)
        lineage: list[dict[str, Any]] = []
        while raw.get("version") == "calibrated-population-query-v1":
            lineage.append(raw)
            raw = raw["source"]
        source: CalibratedPopulationGraph | CalibratedPopulationQuery = (
            calibrated_population_from_dict(raw, limits=cap)
        )
        for item in reversed(lineage):
            source = CalibratedPopulationQuery(
                source,
                recursive_population_graph_from_dict(item["original"], limits=cap),
                item["engine"],
                item["construction_seed"],
                item["version"],
            )
        if type(source) is not CalibratedPopulationQuery:
            raise ValueError("calibrated query archive requires its explicit version")
        source.validate(limits=cap)
        if encode(asdict(source)) != encode(data):
            raise ValueError("calibrated query differs from complete canonical archive")
        return source
    except (KeyError, TypeError, AttributeError, IndexError, OverflowError, RecursionError) as exc:
        raise ValueError("invalid complete calibrated query lineage archive") from exc


def calibrated_source_from_dict(
    data: Any, *, limits: CausalLimits
) -> CalibratedPopulationGraph | CalibratedPopulationQuery:
    if (
        type(data) is dict
        and cast(dict[str, Any], data).get("version") == "calibrated-population-query-v1"
    ):
        return calibrated_query_from_dict(data, limits=limits)
    return calibrated_population_from_dict(data, limits=limits)
