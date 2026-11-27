"""Complete population archives with aggregate admission before construction replay."""

from dataclasses import asdict, fields, replace
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
from r0.modules.priors.lib.recursive_population_records import (
    RecursivePopulationRecipe,
    RecursivePopulationRequest,
    population_identity,
)
from r0.modules.priors.lib.representatives import encode

_DEFAULT_LIMITS = CausalLimits()


def population_raw_cost(data: Any, limits: CausalLimits) -> tuple[int, int]:
    admit_function_metadata(data, limits)
    numbers, storage = 0, 0

    def visit(item: Any) -> None:
        nonlocal numbers, storage
        if type(item) is dict:
            mapping = cast(dict[str, Any], item)
            storage += 128 + 64 * len(mapping)
            for key, value in mapping.items():
                visit(key)
                visit(value)
        elif type(item) in (tuple, list):
            sequence = cast(tuple[Any, ...] | list[Any], item)
            storage += 64 + 16 * len(sequence)
            for value in sequence:
                visit(value)
        elif type(item) is str:
            storage += 64 + len(item.encode("utf-8"))
        elif item is not None:
            numbers += 1
            storage += 64
        else:
            storage += 16

    visit(data)
    if storage > limits.working_bytes:
        raise ValueError("population raw container storage exceeds limits")
    return numbers, storage


def complete_population_record(data: Any, cls: type) -> dict[str, Any]:
    if type(data) is not dict or set(cast(dict[str, Any], data)) != {f.name for f in fields(cls)}:
        raise ValueError("population archive requires every exact field")
    return dict(cast(dict[str, Any], data))


def _decode_request(
    data: Any, *, limits: CausalLimits = _DEFAULT_LIMITS
) -> RecursivePopulationRequest:
    from r0.modules.priors.lib.generate import recipe_from_dict

    admit_function_metadata(data, limits)
    raw = complete_population_record(data, RecursivePopulationRequest)
    declared = CausalLimits(**complete_population_record(raw["limits"], CausalLimits))
    effective = recursive_effective_limits(declared, limits)
    plan = recursive_forest_plan_from_dict(raw["plan"])
    raw_limits = RawDrawLimits(**complete_population_record(raw["raw_limits"], RawDrawLimits))
    budget = plan.admit(effective, raw_limits)
    # Reserve retained matrices and forest replay before any nested source restores its RNG.
    if type(raw["ancestors"]) is not dict:
        raise ValueError("population ancestor requires a complete object")
    ancestor_raw = cast(dict[str, Any], raw["ancestors"])
    ancestor_declared = CausalLimits(
        **complete_population_record(ancestor_raw["limits"], CausalLimits)
    )
    effective = recursive_effective_limits(effective, ancestor_declared)
    budget = plan.admit(effective, raw_limits)
    if type(ancestor_raw.get("task")) is not dict:
        raise ValueError("population ancestor task requires a complete object")
    features = cast(dict[str, Any], ancestor_raw["task"]).get("features")
    if type(features) is not int or not 1 <= features <= 256:
        raise ValueError("population observed feature width is invalid")
    if ancestor_raw["task"].get("rows") != plan.populations:
        raise ValueError("population row ownership differs from ancestor task")
    parents = tuple(tuple(fitted_array(p, 2)) for p in fitted_array(raw["parents"], 256))
    if (
        len(parents) != plan.input_width
        or any(
            len(p) != 2 or any(type(v) is not int for v in p) or p[0] < 0 or p[1] < 0
            for p in parents
        )
        or len(set(parents)) != len(parents)
        or type(raw["target_channel"]) is not int
        or not 0 <= raw["target_channel"] < plan.output_width
    ):
        raise ValueError("invalid population parent slices or target before replay")
    numbers, storage = population_raw_cost(data, effective)
    extra = (
        budget["estimated_working_bytes"]
        + 6 * storage
        + plan.populations
        * (1024 * (plan.input_width + features + plan.output_width) + 2048 * plan.estimators)
    )
    if extra >= effective.working_bytes or budget["parameters"] + numbers >= effective.parameters:
        raise ValueError("population aggregate decode reservation exceeds limits")
    ancestor_limits = replace(
        effective,
        working_bytes=effective.working_bytes - extra,
        parameters=effective.parameters - budget["parameters"] - numbers,
    )
    ancestor_preview = {**ancestor_raw, "limits": asdict(ancestor_limits)}
    ancestors = recipe_from_dict(ancestor_preview, limits=ancestor_limits)
    if type(ancestors) not in ANCESTOR_TYPES:
        raise ValueError("population ancestors require an authored admitted DAG")
    ancestors = replace(
        cast(PopulationAncestors, ancestors),
        limits=ancestor_declared,
    )
    result = RecursivePopulationRequest(
        ancestors,
        parents,
        plan,
        tuple(fitted_array(raw["vocabulary"], 100)),
        raw["target_channel"],
        declared,
        raw_limits,
        raw["version"],
    )
    result.admit(limits=effective)
    if encode(asdict(result)) != encode(data):
        raise ValueError("population request differs from complete canonical archive")
    return result


def _decode_population(
    data: Any, *, limits: CausalLimits = _DEFAULT_LIMITS
) -> RecursivePopulationRecipe:
    admit_function_metadata(data, limits)
    raw = complete_population_record(data, RecursivePopulationRecipe)
    population_identity(raw["seed"], raw["task_id"])
    request = recursive_population_request_from_dict(raw["request"], limits=limits)
    effective = recursive_effective_limits(request.limits, limits)
    matrices: list[tuple[tuple[float, ...], ...]] = []
    for key, width in (
        ("parent_rows", request.plan.input_width),
        ("observed_rows", request.task.features),
    ):
        matrices.append(
            tuple(
                tuple(fitted_array(row, width)) for row in fitted_array(raw[key], request.task.rows)
            )
        )
    sample = sampled_recursive_forest_from_dict(raw["sample"], limits=effective)
    result = RecursivePopulationRecipe(
        request, sample, matrices[0], matrices[1], raw["seed"], raw["task_id"], raw["version"]
    )
    result.validate_replay(limits=effective)
    if encode(asdict(result)) != encode(data):
        raise ValueError("population realization differs from complete canonical archive")
    return result


def _decode_intervention(data: Any, *, limits: CausalLimits = _DEFAULT_LIMITS):
    from r0.modules.priors.lib.causal_intervention_decoding import intervention_from_dict
    from r0.modules.priors.lib.causal_intervention_records import CausalInterventionRecipe
    from r0.modules.priors.lib.recursive_population_intervention import (
        RecursivePopulationIntervention,
    )

    admit_function_metadata(data, limits)
    raw = complete_population_record(data, RecursivePopulationIntervention)
    base_raw = complete_population_record(raw["base"], RecursivePopulationRecipe)
    request_raw = complete_population_record(base_raw["request"], RecursivePopulationRequest)
    plan = recursive_forest_plan_from_dict(request_raw["plan"])
    raw_limits = RawDrawLimits(
        **complete_population_record(request_raw["raw_limits"], RawDrawLimits)
    )
    budget = plan.admit(limits, raw_limits)
    numbers, storage = population_raw_cost(data, limits)
    reserved = budget["estimated_working_bytes"] + 8 * storage
    if reserved >= limits.working_bytes or numbers + budget["parameters"] >= limits.parameters:
        raise ValueError("population intervention decode custody exceeds limits before replay")
    nested_limits = replace(
        limits,
        working_bytes=limits.working_bytes - reserved,
        parameters=limits.parameters - numbers - budget["parameters"],
    )
    base = recursive_population_from_dict(raw["base"], limits=nested_limits)
    ancestors = None
    if raw["ancestors"] is not None:
        ancestor_change = complete_population_record(raw["ancestors"], CausalInterventionRecipe)
        ancestor_change["base"] = {**ancestor_change["base"], "limits": asdict(nested_limits)}
        ancestors = intervention_from_dict(ancestor_change, limits=nested_limits)
    if ancestors is not None:
        ancestors = replace(ancestors, base=base.request.ancestors)
    sample = sampled_recursive_forest_from_dict(raw["sample"], limits=nested_limits)
    parent_rows = tuple(
        tuple(fitted_array(row, base.request.plan.input_width))
        for row in fitted_array(raw["parent_rows"], base.task.rows)
    )
    observed_rows = tuple(
        tuple(fitted_array(row, base.task.features))
        for row in fitted_array(raw["observed_rows"], base.task.rows)
    )
    result = RecursivePopulationIntervention(
        base,
        ancestors,
        raw["mode"],
        raw["population"],
        sample,
        parent_rows,
        observed_rows,
        raw["forced_value"],
        raw["forced_category"],
        raw["version"],
    )
    if encode(asdict(result)) != encode(data):
        raise ValueError("population intervention differs from complete canonical archive")
    return result


def recursive_population_request_from_dict(
    data: Any, *, limits: CausalLimits = _DEFAULT_LIMITS
) -> RecursivePopulationRequest:
    try:
        return _decode_request(data, limits=limits)
    except (KeyError, TypeError, AttributeError, OverflowError, RecursionError) as exc:
        raise ValueError("invalid complete population request archive") from exc


def recursive_population_from_dict(
    data: Any, *, limits: CausalLimits = _DEFAULT_LIMITS
) -> RecursivePopulationRecipe:
    try:
        return _decode_population(data, limits=limits)
    except (KeyError, TypeError, AttributeError, OverflowError, RecursionError) as exc:
        raise ValueError("invalid complete population realization archive") from exc


def recursive_population_intervention_from_dict(
    data: Any, *, limits: CausalLimits = _DEFAULT_LIMITS
):
    try:
        return _decode_intervention(data, limits=limits)
    except (KeyError, TypeError, AttributeError, OverflowError, RecursionError) as exc:
        raise ValueError("invalid complete population intervention archive") from exc
