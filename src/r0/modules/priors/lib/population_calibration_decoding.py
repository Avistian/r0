"""Complete independent-calibration envelope admission before either population replay."""

import math
from dataclasses import asdict
from typing import Any

from r0.modules.priors.lib.causal_limits import CausalLimits
from r0.modules.priors.lib.fitted_fit_decoding import decode_fitted_raw_draw
from r0.modules.priors.lib.function_metadata import admit_function_metadata
from r0.modules.priors.lib.functional_causal import FunctionalCausalRecipe
from r0.modules.priors.lib.functional_target_calibration import target_calibration_from_dict
from r0.modules.priors.lib.generator_snapshot import GeneratorSnapshot
from r0.modules.priors.lib.population_graph_semantic_preflight import (
    authored_semantic_preview,
    complete_version,
)
from r0.modules.priors.lib.population_target_calibration import CalibratedPopulationGraph
from r0.modules.priors.lib.recursive_forest_decoding import recursive_forest_from_dict
from r0.modules.priors.lib.recursive_forest_sampling import SampledRecursiveForest
from r0.modules.priors.lib.recursive_population_decoding import (
    complete_population_record,
    population_raw_cost,
)
from r0.modules.priors.lib.recursive_population_graph_decoding import (
    admit_population_graph_request,
    recursive_population_graph_from_dict,
    recursive_population_graph_request_from_dict,
)
from r0.modules.priors.lib.recursive_population_graph_records import (
    RecursiveNodePopulation,
    RecursivePopulationGraph,
)
from r0.modules.priors.lib.recursive_population_records import population_identity
from r0.modules.priors.lib.recursive_task_preflight import recursive_authored_preview
from r0.modules.priors.lib.representatives import encode, record_fingerprint

_DEFAULT_LIMITS = CausalLimits()


def _population_metadata(data: Any, limits: CausalLimits):
    raw = complete_version(data, RecursivePopulationGraph)
    population_identity(raw["seed"], raw["task_id"])
    request_raw, _, _, _, effective, plans = admit_population_graph_request(raw["request"], limits)
    preview = authored_semantic_preview(recursive_authored_preview(request_raw))
    request = recursive_population_graph_request_from_dict(preview, limits=effective)
    if type(raw["bindings"]) not in (tuple, list) or len(raw["bindings"]) != len(plans):
        raise ValueError("calibration population must retain all node bindings")
    previous_final = None
    for binding, plan in zip(raw["bindings"], plans, strict=True):
        entry = complete_version(binding, RecursiveNodePopulation)
        if type(entry["node"]) is not int or entry["node"] != plan.node:
            raise ValueError("calibration binding node differs")
        values = entry["inputs"]
        if type(values) not in (tuple, list) or len(values) != request.task.rows:
            raise ValueError("calibration binding row count differs")
        for row in values:
            if (
                type(row) not in (tuple, list)
                or len(row) != plan.forest.input_width
                or any(type(v) is not float or not math.isfinite(v) for v in row)
            ):
                raise ValueError("calibration binding input geometry differs")
        sample = complete_version(entry["sample"], SampledRecursiveForest)
        if encode(sample["plan"]) != encode(asdict(plan.forest)):
            raise ValueError("calibration forest plan binding differs")
        if sample["reconstruction"] != "path-local-depth-first-left-first-tree-local-cycle-v1":
            raise ValueError("calibration forest reconstruction differs")
        if any(encode(sample[key]) != encode(request_raw[key]) for key in ("limits", "raw_limits")):
            raise ValueError("calibration forest limits binding differs")
        budget = plan.forest.admit(effective, request.raw_limits)
        forests = sample["forests"]
        if type(forests) not in (tuple, list) or len(forests) != request.task.rows:
            raise ValueError("calibration forest population count differs")
        for item in forests:
            forest = recursive_forest_from_dict(item, limits=effective)
            if (
                forest.input_width != plan.forest.input_width
                or forest.output_width != plan.forest.output_width
                or forest.task != plan.forest.task
                or forest.classes != plan.forest.classes
                or forest.tie_break != plan.forest.tie_break
                or len(forest.trees) != plan.forest.estimators
            ):
                raise ValueError("calibration forest geometry differs")
        snapshots = [
            GeneratorSnapshot(**complete_population_record(sample[key], GeneratorSnapshot))
            for key in ("initial", "final")
        ]
        for snapshot in snapshots:
            snapshot.validate_metadata(byte_limit=request.raw_limits.state_bytes)
        if previous_final is not None and encode(previous_final) != encode(sample["initial"]):
            raise ValueError("calibration per-node forest construction streams are not contiguous")
        previous_final = sample["final"]
        records = sample["raw"]
        if type(records) not in (list, tuple) or len(records) > budget["draws"]:
            raise ValueError("calibration raw record count differs")
        previous, numbers = snapshots[0].digest, 0
        for index, item in enumerate(records):
            record = decode_fitted_raw_draw(item)
            if (
                record.sequence != index
                or record.stream != "raw"
                or record.state_before != previous
                or record.generator != snapshots[0].engine
            ):
                raise ValueError("calibration raw stream chain differs")
            previous = record.state_after
            numbers += len(record.values)
        if (
            numbers > budget["numbers"]
            or previous != snapshots[1].digest
            or snapshots[0].engine != snapshots[1].engine
        ):
            raise ValueError("calibration final stream binding differs")
    return raw, request


def preflight_calibrated_population(data: Any, *, limits: CausalLimits):
    admit_function_metadata(data, limits)
    raw = complete_version(data, CalibratedPopulationGraph)
    declared = CausalLimits(**complete_population_record(raw["limits"], CausalLimits))
    from r0.modules.priors.lib.recursive_forest_plan import recursive_effective_limits

    cap = recursive_effective_limits(declared, limits)
    numbers, storage = population_raw_cost(data, cap)
    if numbers >= cap.parameters or 8 * storage >= cap.working_bytes:
        raise ValueError("complete calibrated population metadata exceeds caller limits")
    from r0.modules.priors.lib.population_calibration_custody import preflight_calibration_custody

    preflight_calibration_custody(data, cap)
    original, query = _population_metadata(raw["original"], cap)
    fit, fit_request = _population_metadata(raw["calibration_population"], cap)
    if not isinstance(query.base, FunctionalCausalRecipe) and query.task.task_type != "regression":
        raise ValueError("direct category targets cannot be numerically calibrated")
    width = len(query.base.target) if isinstance(query.base, FunctionalCausalRecipe) else 1
    calibration = target_calibration_from_dict(raw["calibration"], width)
    if any(
        type(v) is not float for v in (*calibration.center, *calibration.scale)
    ) or calibration.prefix_fingerprint != record_fingerprint(original["request"]):
        raise ValueError("calibration constants or original prefix custody differ")
    expected = {
        **original["request"],
        "base": {
            **original["request"]["base"],
            "task": {
                **original["request"]["base"]["task"],
                "rows": calibration.rows,
                "context_rows": calibration.rows // 2,
            },
        },
        "replacements": [
            {**item, "forest": {**item["forest"], "populations": calibration.rows}}
            for item in original["request"]["replacements"]
        ],
    }
    if encode(fit["request"]) != encode(expected) or (fit["seed"], fit["task_id"]) == (
        original["seed"],
        original["task_id"],
    ):
        raise ValueError("independent calibration population request or identity differs")
    reports = [r.admit(limits=cap) for r in (query, fit_request)]
    if (
        4 * sum(r["parameters"] for r in reports) + numbers > cap.parameters
        or 4 * sum(r["estimated_working_bytes"] for r in reports) + 8 * storage > cap.working_bytes
    ):
        raise ValueError("aggregate retained calibration replay exceeds caller limits")
    from r0.modules.priors.lib.population_calibration_identity import validate_calibration_identity

    validate_calibration_identity(fit)
    return raw, calibration, cap


def calibrated_population_from_dict(data: Any, *, limits: CausalLimits = _DEFAULT_LIMITS):
    try:
        raw, calibration, cap = preflight_calibrated_population(data, limits=limits)
        result = CalibratedPopulationGraph(
            recursive_population_graph_from_dict(raw["original"], limits=cap),
            recursive_population_graph_from_dict(raw["calibration_population"], limits=cap),
            calibration,
            CausalLimits(**raw["limits"]),
            raw["version"],
        )
        result.validate()
        if encode(asdict(result)) != encode(data):
            raise ValueError("calibrated population differs from complete canonical archive")
        return result
    except (KeyError, TypeError, AttributeError, IndexError, RecursionError, OverflowError) as exc:
        raise ValueError("invalid complete calibrated population archive") from exc
