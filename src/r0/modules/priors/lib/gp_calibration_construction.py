"""Admit independent fitting controls and future custody before decoding GP sources."""

from dataclasses import fields, replace
from typing import Any

from r0.modules.priors.lib.causal_limits import CausalLimits
from r0.modules.priors.lib.functional_causal import FunctionalCausalRecipe
from r0.modules.priors.lib.gp_root_calibration import calibrate_gp_root_dag
from r0.modules.priors.lib.gp_root_calibration_admission import admit_gp_calibration_metadata
from r0.modules.priors.lib.gp_root_decoding import gp_root_dag_from_dict, preflight_gp_root_dag


def preflight_gp_calibration_construction(
    data: dict[str, Any], *, seed: int, task_id: int, limits: CausalLimits
):
    if set(data) != {
        "source",
        "calibration_rows",
        "calibration_seed",
        "calibration_task_id",
        "calibration_locations",
    }:
        raise ValueError("GP fitting requires complete source and independent population controls")
    if any(type(v) is not int or not 0 <= v < 2**63 for v in (seed, task_id)):
        raise ValueError("GP fitting requires exact query identity")
    source = data["source"]
    cap, original, bindings, request = preflight_gp_root_dag(source, limits=limits)
    if (source["source"]["seed"], source["source"]["task_id"]) != (seed, task_id):
        raise ValueError("GP fitting query identity differs from its retained source")
    fit_seed, fit_id, rows = (
        data["calibration_seed"],
        data["calibration_task_id"],
        data["calibration_rows"],
    )
    if any(type(v) is not int or not 0 <= v < 2**63 for v in (fit_seed, fit_id)):
        raise ValueError("GP fitting requires exact independent population identity")
    if (fit_seed, fit_id) == (seed, task_id):
        raise ValueError("GP fitting requires an independent population identity")
    if type(rows) is not int or not 32 <= rows <= 1024:
        raise ValueError("GP fitting requires 32..1024 independent rows")
    if not isinstance(original, FunctionalCausalRecipe) and original.task.task_type != "regression":
        raise ValueError("direct categorical targets require category-preserving calibration")
    if isinstance(original, FunctionalCausalRecipe) and original.target_calibration_rows:
        raise ValueError("GP fitting cannot substitute placeholder iid calibration")
    declared = CausalLimits(**source["base"]["limits"])
    if any(getattr(declared, f.name) > getattr(cap, f.name) for f in fields(CausalLimits)):
        raise ValueError("GP fitting graph caps exceed retained replay limits")
    locations = data["calibration_locations"]
    if locations is not None:
        if type(locations) not in (tuple, list) or any(
            type(row) not in (tuple, list) for row in locations
        ):
            raise ValueError("GP calibration locations require a numeric matrix")
        locations = tuple(tuple(row) for row in locations)
    fit_request = replace(
        request, rows=rows, locations=request.locations if locations is None else locations
    )
    base = replace(original, task=replace(original.task, rows=rows, context_rows=rows // 2))
    fit_metadata = {
        **source["base"],
        "task": {**source["base"]["task"], "rows": rows, "context_rows": rows // 2},
    }
    admit_gp_calibration_metadata(
        source,
        original_limits=cap,
        original_base=original,
        original_request=request,
        bindings=bindings,
        request=fit_request,
        base=base,
        limits=limits,
        fit_metadata=fit_metadata,
    )
    return rows, fit_seed, fit_id, locations


def construct_gp_root_calibration(
    data: dict[str, Any], *, seed: int, task_id: int, limits: CausalLimits
):
    rows, fit_seed, fit_id, locations = preflight_gp_calibration_construction(
        data, seed=seed, task_id=task_id, limits=limits
    )
    decoded = gp_root_dag_from_dict(data["source"], limits=limits)
    return calibrate_gp_root_dag(
        decoded,
        calibration_rows=rows,
        calibration_seed=fit_seed,
        calibration_task_id=fit_id,
        calibration_locations=locations,
        limits=limits,
    )


def gp_calibration_construction_matches(
    controls: dict[str, Any],
    record: dict[str, Any],
    *,
    seed: int,
    task_id: int,
    limits: CausalLimits,
) -> bool:
    from r0.modules.priors.lib.gp_column_decoding import gp_payload_equal
    from r0.modules.priors.lib.gp_root_calibration_decoding import preflight_gp_root_calibration
    from r0.modules.priors.lib.gp_task_validation import validate_gp_execution

    try:
        if set(record) != {"recipe", "seed", "task_id", "execution"} or any(
            type(record.get(k)) is not int or record[k] != value
            for k, value in (("seed", seed), ("task_id", task_id))
        ):
            return False
        validate_gp_execution(record["execution"], "gp-root-calibration-v1")
        raw = record["recipe"]
        if not gp_payload_equal(raw["original"], controls["source"]):
            return False
        rows, fit_seed, fit_id, locations = preflight_gp_calibration_construction(
            {k: v for k, v in controls.items() if k != "topology"},
            seed=seed,
            task_id=task_id,
            limits=limits,
        )
        source = raw["fit"]["source"]
        expected_request = {
            **controls["source"]["source"]["request"],
            "rows": rows,
        }
        if locations is not None:
            expected_request["locations"] = locations
        if (
            type(source["seed"]) is not int
            or source["seed"] != fit_seed
            or type(source["task_id"]) is not int
            or source["task_id"] != fit_id
            or not gp_payload_equal(source["request"], expected_request)
        ):
            return False
        preflight_gp_root_calibration(raw, limits=limits)
        return True
    except (KeyError, TypeError, AttributeError, IndexError, ValueError, OverflowError):
        return False
