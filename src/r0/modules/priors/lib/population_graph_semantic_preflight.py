"""Validate population comparison semantics using non-replaying authored previews."""

import math
from dataclasses import asdict, fields
from typing import Any, cast

from r0.modules.priors.lib.causal_intervention_decoding import intervention_from_dict
from r0.modules.priors.lib.causal_limits import CausalLimits
from r0.modules.priors.lib.fitted_tree_decoding import fitted_array
from r0.modules.priors.lib.recursive_forest_sampling import SampledRecursiveForest
from r0.modules.priors.lib.recursive_population_decoding import complete_population_record
from r0.modules.priors.lib.recursive_population_graph_plan import RecursivePopulationGraphRequest
from r0.modules.priors.lib.recursive_population_graph_records import (
    RecursiveNodePopulation,
    RecursivePopulationGraph,
)
from r0.modules.priors.lib.recursive_population_records import population_identity
from r0.modules.priors.lib.representatives import encode


def complete_version(data: Any, cls: type[Any]) -> dict[str, Any]:
    raw = complete_population_record(data, cls)
    version = next(field.default for field in fields(cls) if field.name == "version")
    if type(raw["version"]) is not str or raw["version"] != version:
        raise ValueError("population envelope requires its explicit version")
    return raw


def authored_semantic_preview(value: Any) -> Any:
    """Keep realized functions/laws, omit replay wrappers only in this temporary view."""
    if type(value) in (list, tuple):
        return tuple(authored_semantic_preview(v) for v in cast(list[Any] | tuple[Any, ...], value))
    if type(value) is not dict:
        return value
    raw = cast(dict[str, Any], value)
    if raw.get("version") == "frozen-recursive-forest-function-v1":
        source = raw["source"]
        if type(source) is not dict:
            raise ValueError("population semantic preview requires a forest source object")
        source = cast(dict[str, Any], source)
        if source.get("version") == "sampled-recursive-forest-v1":
            if source["plan"]["lifecycle"] != "frozen-function" or len(source["forests"]) != 1:
                raise ValueError("population semantic preview requires one frozen source forest")
            return {**raw, "source": authored_semantic_preview(source["forests"][0])}
    if raw.get("version") == "sampled-source-root-base-v1":
        return {
            "version": "source-root-base-v1",
            "law": authored_semantic_preview(raw["law"]),
            "calibration": authored_semantic_preview(raw["calibration"]),
        }
    if raw.get("version") == "prototype-root-base-v1":
        bank = raw["bank"]
        if type(bank) is not dict:
            raise ValueError("population semantic preview requires a prototype bank object")
        bank = cast(dict[str, Any], bank)
        if bank.get("version") == "sampled-prototype-root-v1":
            return {**raw, "bank": authored_semantic_preview(bank["root"])}
    return {key: authored_semantic_preview(child) for key, child in raw.items()}


def _matrix(data: Any, rows: int, width: int) -> None:
    values = fitted_array(data, rows)
    if len(values) != rows:
        raise ValueError("population outer input row count differs")
    for row in values:
        entries = fitted_array(row, width)
        if len(entries) != width or any(
            type(v) is not float or not math.isfinite(v) for v in entries
        ):
            raise ValueError("population outer input geometry or scalar type differs")


def _receipt(data: Any, plan: dict[str, Any], request: dict[str, Any]) -> None:
    sample = complete_version(data, SampledRecursiveForest)
    if any(
        encode(sample[key]) != encode(value)
        for key, value in (
            ("plan", plan),
            ("limits", request["limits"]),
            ("raw_limits", request["raw_limits"]),
        )
    ):
        raise ValueError("population outer receipt plan or limits differ")


def preflight_population_change(
    raw: dict[str, Any], baseline: dict[str, Any], change: dict[str, Any], *, limits: CausalLimits
) -> None:
    """Original packet remains authoritative; previews are discarded after validation."""
    from r0.modules.priors.lib.recursive_population_graph_intervention import (
        RecursivePopulationGraphIntervention,
    )

    complete_version(raw, RecursivePopulationGraphIntervention)
    baseline = complete_version(baseline, RecursivePopulationGraph)
    population_identity(baseline["seed"], baseline["task_id"])
    request = complete_version(baseline["request"], RecursivePopulationGraphRequest)
    # This invokes the SAME intervention constructors and graph compatibility checks
    # as actual decoding, without restoring retained function/root constructions.
    preview = authored_semantic_preview(change)
    preview["base"] = {**preview["base"], "limits": asdict(limits)}
    validated_change = intervention_from_dict(preview, limits=limits)
    plans = fitted_array(request["replacements"], limits.nodes)
    from r0.modules.priors.lib.causal_intervention_records import SoftIntervention
    from r0.modules.priors.lib.recursive_forest_decoding import recursive_forest_plan_from_dict
    from r0.modules.priors.lib.recursive_population_graph_plan import (
        RecursiveNodePopulationPlan,
        validate_population_emission,
    )

    runtime = validated_change.effective_runtime()
    disabled = {
        c.node
        for c in validated_change.interventions
        if isinstance(c, SoftIntervention) and c.function is not None
    }
    for plan in plans:
        binding = complete_version(plan, RecursiveNodePopulationPlan)
        typed = RecursiveNodePopulationPlan(
            binding["node"], recursive_forest_plan_from_dict(binding["forest"]), binding["version"]
        )
        if typed.node not in disabled:
            validate_population_emission(runtime, typed)
    rows = request["base"]["task"]["rows"]
    bindings = fitted_array(baseline["bindings"], len(plans))
    samples = fitted_array(raw["samples"], len(plans))
    inputs = fitted_array(raw["inputs"], len(plans))
    if any(len(items) != len(plans) for items in (bindings, samples, inputs)):
        raise ValueError("population comparison outer receipt/input count differs")
    for binding, sample, pair, plan in zip(bindings, samples, inputs, plans, strict=True):
        entry = complete_version(binding, RecursiveNodePopulation)
        node = plan["node"]
        if type(entry["node"]) is not int or entry["node"] != node:
            raise ValueError("population baseline outer node binding differs")
        _matrix(entry["inputs"], rows, plan["forest"]["input_width"])
        _receipt(entry["sample"], plan["forest"], request)
        _receipt(sample, plan["forest"], request)
        address = fitted_array(pair, 2)
        if len(address) != 2 or type(address[0]) is not int or address[0] != node:
            raise ValueError("population comparison outer input node order differs")
        if address[1] is not None:
            _matrix(address[1], rows, plan["forest"]["input_width"])
