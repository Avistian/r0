"""Array-free whole-population graph reservation precedes any nested receipt replay."""

from dataclasses import asdict, replace
from typing import Any, cast

from r0.modules.priors.lib.causal_limits import CausalLimits
from r0.modules.priors.lib.fitted_tree_decoding import fitted_array
from r0.modules.priors.lib.function_metadata import admit_function_metadata
from r0.modules.priors.lib.raw_draw_records import RawDrawLimits
from r0.modules.priors.lib.recursive_forest_decoding import (
    recursive_forest_plan_from_dict,
    sampled_recursive_forest_from_dict,
)
from r0.modules.priors.lib.recursive_forest_plan import recursive_effective_limits
from r0.modules.priors.lib.recursive_population_ancestors import ANCESTOR_TYPES, PopulationAncestors
from r0.modules.priors.lib.recursive_population_decoding import (
    complete_population_record,
    population_raw_cost,
)
from r0.modules.priors.lib.recursive_population_graph_plan import (
    RecursiveNodePopulationPlan,
    RecursivePopulationGraphRequest,
)
from r0.modules.priors.lib.recursive_population_graph_records import (
    RecursiveNodePopulation,
    RecursivePopulationGraph,
)
from r0.modules.priors.lib.recursive_population_records import population_identity
from r0.modules.priors.lib.recursive_task_preflight import admit_recursive_tasks
from r0.modules.priors.lib.representatives import encode

_DEFAULT_LIMITS = CausalLimits()


def admit_population_graph_request(data: Any, limits: CausalLimits):
    admit_function_metadata(data, limits)
    raw = complete_population_record(data, RecursivePopulationGraphRequest)
    declared = CausalLimits(**complete_population_record(raw["limits"], CausalLimits))
    raw_limits = RawDrawLimits(**complete_population_record(raw["raw_limits"], RawDrawLimits))
    if type(raw["base"]) is not dict:
        raise ValueError("population graph base must be an explicit authored DAG")
    base_raw = cast(dict[str, Any], raw["base"])
    base_limits = CausalLimits(**complete_population_record(base_raw["limits"], CausalLimits))
    effective = recursive_effective_limits(
        recursive_effective_limits(declared, limits), base_limits
    )
    plans: list[RecursiveNodePopulationPlan] = []
    for item in fitted_array(raw["replacements"], effective.nodes):
        entry = complete_population_record(item, RecursiveNodePopulationPlan)
        plans.append(
            RecursiveNodePopulationPlan(
                entry["node"], recursive_forest_plan_from_dict(entry["forest"]), entry["version"]
            )
        )
    if not plans or tuple(p.node for p in plans) != tuple(sorted({p.node for p in plans})):
        raise ValueError("population graph replacements must be unique and ordered")
    nodes = fitted_array(base_raw["nodes"], effective.nodes)
    if type(base_raw.get("task")) is not dict:
        raise ValueError("population graph task must be complete")
    rows = cast(dict[str, Any], base_raw["task"]).get("rows")
    if type(rows) is not int or any(
        p.node >= len(nodes) or p.forest.populations != rows for p in plans
    ):
        raise ValueError("population graph node or row ownership differs before replay")
    reports = [p.forest.admit(effective, raw_limits) for p in plans]
    if (
        len(nodes) + sum(r["nodes"] for r in reports) > effective.nodes
        or sum(r["draws"] for r in reports) > raw_limits.draws
        or sum(r["numbers"] for r in reports) > raw_limits.numbers
    ):
        raise ValueError("aggregate population graph raw or node caps exceeded before replay")
    numbers, storage = population_raw_cost(data, effective)
    replay = sum(r["estimated_working_bytes"] for r in reports)
    retained = rows * sum(
        p.forest.input_width + (p.forest.classes or p.forest.output_width) for p in plans
    )
    replay += 2048 * retained
    numbers += sum(r["parameters"] for r in reports) + 8 * retained
    if numbers > effective.parameters or replay + 8 * storage > effective.working_bytes:
        raise ValueError("population graph aggregate reservation exceeds limits before replay")
    admit_recursive_tasks(
        [base_raw],
        storage=storage,
        numbers=numbers,
        replay_work=replay,
        limits=effective,
        retain_node_stages=True,
        population_geometry={
            p.node: (
                p.forest.input_width,
                p.forest.classes or p.forest.output_width,
                p.forest.classes,
            )
            for p in plans
        },
    )
    return raw, base_raw, declared, raw_limits, effective, plans


def _decode_request(data: Any, limits: CausalLimits) -> RecursivePopulationGraphRequest:
    from r0.modules.priors.lib.generate import recipe_from_dict

    raw, base_raw, declared, raw_limits, effective, plans = admit_population_graph_request(
        data, limits
    )
    preview = {**base_raw, "limits": asdict(effective)}
    base = recipe_from_dict(preview, limits=effective)
    if type(base) not in ANCESTOR_TYPES:
        raise ValueError("population graph requires an ordinary authored four-family base")
    base = replace(cast(PopulationAncestors, base), limits=CausalLimits(**base_raw["limits"]))
    result = RecursivePopulationGraphRequest(
        base, tuple(plans), declared, raw_limits, raw["version"]
    )
    result.admit(limits=effective)
    if encode(asdict(result)) != encode(data):
        raise ValueError("population graph request differs from complete canonical archive")
    return result


def recursive_population_graph_request_from_dict(
    data: Any, *, limits: CausalLimits = _DEFAULT_LIMITS
) -> RecursivePopulationGraphRequest:
    try:
        return _decode_request(data, limits)
    except (KeyError, TypeError, AttributeError, OverflowError, RecursionError) as exc:
        raise ValueError("invalid complete population graph request") from exc


def recursive_population_graph_from_dict(
    data: Any, *, limits: CausalLimits = _DEFAULT_LIMITS
) -> RecursivePopulationGraph:
    try:
        admit_function_metadata(data, limits)
        raw = complete_population_record(data, RecursivePopulationGraph)
        population_identity(raw["seed"], raw["task_id"])
        numbers, storage = population_raw_cost(data, limits)
        if numbers >= limits.parameters or 8 * storage >= limits.working_bytes:
            raise ValueError("complete population DAG metadata exceeds caller caps before replay")
        nested = replace(
            limits,
            parameters=limits.parameters - numbers,
            working_bytes=limits.working_bytes - 8 * storage,
        )
        request = recursive_population_graph_request_from_dict(raw["request"], limits=nested)
        effective = recursive_effective_limits(request.limits, limits)
        rows = request.task.rows
        bindings: list[RecursiveNodePopulation] = []
        entries = fitted_array(raw["bindings"], len(request.replacements))
        if len(entries) != len(request.replacements):
            raise ValueError("population graph receipt count differs")
        # Validate all outer geometry before allowing any per-node construction replay.
        parsed = [complete_population_record(entry, RecursiveNodePopulation) for entry in entries]
        for entry, plan in zip(parsed, request.replacements, strict=True):
            if type(entry["node"]) is not int or entry["node"] != plan.node:
                raise ValueError("population graph receipt node differs")
            matrix = fitted_array(entry["inputs"], rows)
            if len(matrix) != rows or any(
                len(fitted_array(row, plan.forest.input_width)) != plan.forest.input_width
                for row in matrix
            ):
                raise ValueError("population graph original input geometry differs")
        for entry in parsed:
            sample = sampled_recursive_forest_from_dict(entry["sample"], limits=effective)
            inputs = tuple(
                tuple(fitted_array(row, sample.plan.input_width))
                for row in fitted_array(entry["inputs"], rows)
            )
            bindings.append(
                RecursiveNodePopulation(entry["node"], sample, inputs, entry["version"])
            )
        result = RecursivePopulationGraph(
            request, tuple(bindings), raw["seed"], raw["task_id"], raw["version"]
        )
        result.validate_replay(limits=effective)
        if encode(asdict(result)) != encode(data):
            raise ValueError("population graph differs from complete canonical archive")
        return result
    except (KeyError, TypeError, AttributeError, OverflowError, RecursionError) as exc:
        raise ValueError("invalid complete population graph archive") from exc
