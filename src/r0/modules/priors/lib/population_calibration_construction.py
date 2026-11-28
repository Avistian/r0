"""Guarded raw construction admits all retained and proposed populations before replay."""

from dataclasses import asdict
from typing import Any

from r0.modules.priors.lib.calibrated_population_query import (
    realize_calibrated_population_query,
)
from r0.modules.priors.lib.calibrated_query_admission import preflight_calibrated_query
from r0.modules.priors.lib.calibrated_query_decoding import (
    admit_query_aggregate,
    calibrated_source_from_dict,
)
from r0.modules.priors.lib.causal_limits import CausalLimits
from r0.modules.priors.lib.function_metadata import admit_function_metadata
from r0.modules.priors.lib.population_calibration_custody import preflight_calibration_custody
from r0.modules.priors.lib.population_calibration_decoding import calibration_population_metadata
from r0.modules.priors.lib.population_target_calibration import (
    calibrate_population_graph,
    calibration_request_from_source,
)
from r0.modules.priors.lib.recursive_population_decoding import population_raw_cost
from r0.modules.priors.lib.recursive_population_graph_decoding import (
    recursive_population_graph_from_dict,
)
from r0.modules.priors.lib.recursive_population_records import population_identity


def construct_calibrated_population(
    controls: dict[str, Any],
    *,
    seed: int,
    task_id: int,
    construction_seed: int,
    limits: CausalLimits,
):
    """Fit explicitly or create one same-law query; never decode before outer admission."""
    admit_function_metadata(controls, limits)
    population_identity(seed, task_id)
    population_identity(construction_seed, task_id)
    try:
        topology = controls["topology"]
        source = controls["source"]
        preflight_calibration_custody(source, limits)
        if topology == "calibrate-population":
            if set(controls) != {
                "topology",
                "source",
                "calibration_rows",
                "calibration_seed",
                "calibration_task_id",
            }:
                raise ValueError("calibration construction requires complete explicit controls")
            fit_seed, fit_id = controls["calibration_seed"], controls["calibration_task_id"]
            population_identity(fit_seed, fit_id)
            raw, request = calibration_population_metadata(source, limits)
            if (seed, task_id) != (raw["seed"], raw["task_id"]):
                raise ValueError("calibration preserves the original query identity")
            if (seed, task_id) == (fit_seed, fit_id):
                raise ValueError("calibration identity must be independent")
            if construction_seed != fit_seed:
                raise ValueError("calibration construction seed must equal its recorded fit seed")
            fit_request = calibration_request_from_source(request, controls["calibration_rows"])
            numbers, storage = population_raw_cost((source, asdict(fit_request)), limits)
            admit_query_aggregate(
                [request.admit(limits=limits), fit_request.admit(limits=limits)],
                numbers,
                storage,
                limits,
            )
            original = recursive_population_graph_from_dict(source, limits=limits)
            return calibrate_population_graph(
                original,
                calibration_rows=controls["calibration_rows"],
                calibration_seed=fit_seed,
                calibration_task_id=fit_id,
                limits=limits,
            )
        if topology != "calibrated-population-query" or set(controls) != {
            "topology",
            "source",
            "engine",
        }:
            raise ValueError("query construction requires complete retained source and engine")
        cap = preflight_calibrated_query(
            source,
            seed=seed,
            task_id=task_id,
            engine=controls["engine"],
            construction_seed=construction_seed,
            limits=limits,
        )
        original_source = calibrated_source_from_dict(source, limits=cap)
        return realize_calibrated_population_query(
            original_source,
            seed=seed,
            task_id=task_id,
            engine=controls["engine"],
            construction_seed=construction_seed,
        )
    except (KeyError, TypeError, AttributeError, IndexError) as exc:
        raise ValueError("calibrated construction requires complete metadata and controls") from exc
