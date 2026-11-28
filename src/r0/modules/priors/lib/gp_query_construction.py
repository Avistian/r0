"""Guarded query construction admits the full future lineage before source replay."""

from typing import Any

from r0.modules.priors.lib.causal_limits import CausalLimits
from r0.modules.priors.lib.gp_root_calibration_decoding import gp_root_calibration_from_dict
from r0.modules.priors.lib.gp_root_query import realize_gp_root_calibration_query
from r0.modules.priors.lib.gp_root_query_admission import preflight_gp_root_calibration_query
from r0.modules.priors.lib.gp_root_query_decoding import gp_root_calibration_query_from_dict


def construct_gp_root_calibration_query(
    data: dict[str, Any], *, seed: int, task_id: int, limits: CausalLimits
):
    if type(data) is not dict or set(data) != {"source", "locations"}:
        raise ValueError("GP query requires complete source and explicit location controls")
    locations = data["locations"]
    if locations is not None:
        if type(locations) not in (list, tuple) or any(
            type(row) not in (list, tuple) for row in locations
        ):
            raise ValueError("GP query locations require a matrix or null")
        locations = tuple(tuple(row) for row in locations)
    preflight_gp_root_calibration_query(
        data["source"], seed=seed, task_id=task_id, locations=locations, limits=limits
    )
    decoder = (
        gp_root_calibration_query_from_dict
        if data["source"]["version"] == "gp-root-calibration-query-v1"
        else gp_root_calibration_from_dict
    )
    source = decoder(data["source"], limits=limits)
    return realize_gp_root_calibration_query(
        source, seed=seed, task_id=task_id, locations=locations, limits=limits
    )


def gp_query_construction_matches(
    controls: dict[str, Any],
    record: dict[str, Any],
    *,
    seed: int,
    task_id: int,
    limits: CausalLimits,
) -> bool:
    from dataclasses import asdict

    from r0.modules.priors.lib.gp_column_decoding import gp_payload_equal
    from r0.modules.priors.lib.gp_root_query_admission import query_construction_plan
    from r0.modules.priors.lib.gp_root_query_metadata import query_source_metadata
    from r0.modules.priors.lib.gp_task_validation import validate_gp_execution

    try:
        if set(controls) != {"topology", "source", "locations"} or set(record) != {
            "recipe",
            "seed",
            "task_id",
            "execution",
        }:
            return False
        if any(
            type(record[k]) is not int or record[k] != v
            for k, v in (("seed", seed), ("task_id", task_id))
        ):
            return False
        validate_gp_execution(record["execution"], "gp-root-calibration-query-v1")
        raw = record["recipe"]
        if record["execution"]["device"] != "cpu" or not gp_payload_equal(
            raw["source"], controls["source"]
        ):
            return False
        locations = controls["locations"]
        if locations is not None:
            locations = tuple(tuple(row) for row in locations)
        metadata, request = query_construction_plan(
            controls["source"], seed=seed, task_id=task_id, locations=locations, limits=limits
        )
        query_source_metadata(raw, limits=limits)
        current = raw["original"]["source"]
        return (
            gp_payload_equal(raw["limits"], asdict(metadata.limits))
            and gp_payload_equal(current["request"], asdict(request))
            and type(current["seed"]) is int
            and current["seed"] == seed
            and type(current["task_id"]) is int
            and current["task_id"] == task_id
        )
    except (KeyError, TypeError, ValueError, AttributeError, IndexError, OverflowError):
        return False
