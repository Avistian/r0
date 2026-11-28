"""Request-bound validation of archived recipe and construction identities."""

import json
from dataclasses import asdict, replace
from typing import Any, cast

from r0.modules.generation.lib.requests import RunRequest, sampling_record, selected_recipe
from r0.modules.priors import (
    CausalLimits,
    GPRootCalibration,
    GPRootCalibrationQuery,
    GPRootDAGRecipe,
    recipe_from_dict,
)


def record_matches(record: dict[str, Any], request: RunRequest, task_id: int) -> bool:
    if request.observation is None:
        if "observation" in record:
            return False
    elif not request.observation.matches(record, request.causal_limits or CausalLimits()):
        return False
    return _source_matches(
        {k: v for k, v in record.items() if k != "observation"}, request, task_id
    )


def _source_matches(record: dict[str, Any], request: RunRequest, task_id: int) -> bool:
    from r0.modules.priors import (
        PrototypeBatchNetworkRecipe,
        RealizedPrototypeBatchNetworkRecipe,
    )

    if request.construction is not None:
        controls = json.loads(request.construction.controls_json)
        if controls.get("topology") in ("tree-extension", "tree-ensemble"):
            from r0.modules.generation.lib.tree_extension_construction import tree_record_matches

            return tree_record_matches(
                record,
                request.construction,
                limits=request.causal_limits or CausalLimits(),
                seed=request.seed,
                task_id=task_id,
            )
        if controls.get("topology") == "query-calibrated-gp-root":
            from r0.modules.priors import gp_query_construction_matches

            if not gp_query_construction_matches(
                controls,
                record,
                seed=request.seed,
                task_id=task_id,
                limits=request.causal_limits or CausalLimits(),
            ):
                return False
        if controls.get("topology") == "calibrate-gp-root":
            from r0.modules.priors import gp_calibration_construction_matches

            if not gp_calibration_construction_matches(
                controls,
                record,
                seed=request.seed,
                task_id=task_id,
                limits=request.causal_limits or CausalLimits(),
            ):
                return False
        if controls.get("topology") == "xgboost-population":
            from r0.modules.priors import xgboost_population_construction_matches

            return xgboost_population_construction_matches(
                controls,
                record,
                construction_seed=request.construction.seed,
                seed=request.seed,
                task_id=task_id,
                limits=request.causal_limits or CausalLimits(),
            )
        if controls.get("topology") == "gp-root-dag":
            from r0.modules.priors import gp_root_construction_matches

            if not gp_root_construction_matches(
                controls,
                record,
                seed=request.seed,
                task_id=task_id,
                limits=request.causal_limits or CausalLimits(),
            ):
                return False
    selected = selected_recipe(request, task_id)
    from r0.modules.generation.lib.xgboost_population_transport import (
        xgboost_population_record_matches,
    )

    match = xgboost_population_record_matches(record, request, selected, task_id)
    if match is not None:
        return match
    from r0.modules.priors import (
        GPColumnPopulation,
        GPColumnRecipe,
        GPColumnTargetLink,
        GPColumnTargetRecipe,
    )

    if isinstance(selected, (GPRootDAGRecipe, GPRootCalibration, GPRootCalibrationQuery)):
        return (
            set(record) == {"recipe", "seed", "task_id", "execution"}
            and type(record.get("seed")) is int
            and record["seed"] == request.seed
            and type(record.get("task_id")) is int
            and record["task_id"] == task_id
            and json.dumps(record.get("recipe"), sort_keys=True, allow_nan=False)
            == json.dumps(asdict(selected), sort_keys=True, allow_nan=False)
        )
    if isinstance(selected, (GPColumnTargetRecipe, GPColumnTargetLink)):
        if any(
            type(record.get(k)) is not int or record[k] != v
            for k, v in (("seed", request.seed), ("task_id", task_id))
        ):
            return False
        if json.dumps(record.get("recipe"), sort_keys=True, allow_nan=False) != json.dumps(
            asdict(selected), sort_keys=True, allow_nan=False
        ) or record.get("sampling") != sampling_record(request):
            return False
        recipe_from_dict(record["recipe"], limits=request.causal_limits or selected.request.limits)
        return True
    if isinstance(selected, GPColumnRecipe):
        if any(
            type(record.get(key)) is not int or record[key] != identity
            for key, identity in (("seed", request.seed), ("task_id", task_id))
        ):
            return False
        raw = record.get("recipe")
        if type(raw) is not dict:
            return False
        raw = cast(dict[str, Any], raw)
        source = raw.get("source")
        if type(source) is not dict:
            return False
        source = cast(dict[str, Any], source)
        if (
            raw.get("version") != selected.version
            or source.get("version") != "gp-column-population-v1"
            or any(
                type(source.get(key)) is not int or source[key] != identity
                for key, identity in (("seed", request.seed), ("task_id", task_id))
            )
            or record.get("sampling") != sampling_record(request)
        ):
            return False
        # These controls are known without any numerical population replay.
        actual_controls = {"task": raw.get("task"), "request": source.get("request")}
        expected_controls = {"task": asdict(selected.task), "request": asdict(selected.request)}
        if json.dumps(actual_controls, sort_keys=True, allow_nan=False) != json.dumps(
            expected_controls, sort_keys=True, allow_nan=False
        ):
            return False
        if isinstance(selected.source, GPColumnPopulation) and json.dumps(
            raw, sort_keys=True, allow_nan=False
        ) != json.dumps(asdict(selected), sort_keys=True, allow_nan=False):
            return False
        realized = recipe_from_dict(raw, limits=request.causal_limits or selected.request.limits)
        if not isinstance(realized, GPColumnRecipe) or not isinstance(
            realized.source, GPColumnPopulation
        ):
            return False
        if (realized.source.seed, realized.source.task_id) != (request.seed, task_id):
            return False
        expected = (
            selected
            if isinstance(selected.source, GPColumnPopulation)
            else replace(selected, source=realized.source)
        )
        return (
            json.dumps(asdict(realized), sort_keys=True, allow_nan=False)
            == json.dumps(asdict(expected), sort_keys=True, allow_nan=False)
            and json.dumps(asdict(realized.request), sort_keys=True, allow_nan=False)
            == json.dumps(asdict(selected.request), sort_keys=True, allow_nan=False)
            and record.get("sampling") == sampling_record(request)
        )
    if isinstance(selected, (PrototypeBatchNetworkRecipe, RealizedPrototypeBatchNetworkRecipe)):
        # Python numeric equality aliases bool/float identities to integers. Check
        # transport primitives before replay so a persisted record can replay itself.
        raw = record.get("recipe")
        if type(raw) is not dict:
            return False
        raw = cast(dict[str, Any], raw)
        if any(
            type(record.get(key)) is not int
            or type(raw.get(key)) is not int
            or record[key] != expected_id
            or raw[key] != expected_id
            for key, expected_id in (("seed", request.seed), ("task_id", task_id))
        ):
            return False
        realized = recipe_from_dict(
            record["recipe"], limits=request.causal_limits or selected.limits
        )
        expected = (
            selected.request
            if isinstance(selected, RealizedPrototypeBatchNetworkRecipe)
            else selected
        )
        return (
            isinstance(realized, RealizedPrototypeBatchNetworkRecipe)
            and json.dumps(raw.get("request"), sort_keys=True, allow_nan=False)
            == json.dumps(asdict(expected), sort_keys=True, allow_nan=False)
            and (realized.seed, realized.task_id) == (request.seed, task_id)
            and record.get("sampling") == sampling_record(request)
            and record["seed"] == request.seed
            and record["task_id"] == task_id
        )
    return (
        record["recipe"] == json.loads(json.dumps(asdict(selected)))
        and record.get("sampling") == sampling_record(request)
        and record["seed"] == request.seed
        and record["task_id"] == task_id
    )
