"""Known construction metadata binding before any finite GP population replay."""

from dataclasses import asdict
from typing import Any, cast

from r0.modules.priors.lib.causal_limits import CausalLimits
from r0.modules.priors.lib.gp_column_decoding import (
    gp_column_request_from_dict,
    gp_payload_equal,
)
from r0.modules.priors.lib.gp_root_decoding import preflight_gp_root_dag
from r0.modules.priors.lib.population_graph_semantic_preflight import authored_semantic_preview
from r0.modules.priors.lib.recursive_task_preflight import recursive_authored_preview


def _submitted_fields(expected: Any, actual: Any, limits: CausalLimits) -> bool:
    """Bind submitted custody with exact primitive types, before replay."""
    if type(expected) is dict:
        if type(actual) is not dict:
            return False
        expected = cast(dict[str, Any], expected)
        actual = cast(dict[str, Any], actual)
        for key, value in expected.items():
            if key not in actual:
                return False
            if key == "limits" and type(value) is dict and type(actual[key]) is dict:
                ceiling = asdict(limits)
                value = {
                    k: min(v, ceiling[k]) if k in ceiling and type(v) is int else v
                    for k, v in cast(dict[str, Any], value).items()
                }
            if not _submitted_fields(value, actual[key], limits):
                return False
        return True
    if type(expected) in (tuple, list):
        if type(actual) not in (tuple, list) or len(expected) != len(actual):
            return False
        return all(_submitted_fields(a, b, limits) for a, b in zip(expected, actual, strict=True))
    return type(expected) is type(actual) and expected == actual


def gp_root_construction_matches(
    controls: dict[str, Any],
    record: dict[str, Any],
    *,
    seed: int,
    task_id: int,
    limits: CausalLimits,
) -> bool:
    """Cheap known-input binding; full realized population/task validation still follows."""
    from r0.modules.priors.lib.generate import recipe_from_dict
    from r0.modules.priors.lib.gp_task_validation import validate_gp_execution

    try:
        if set(record) != {"recipe", "seed", "task_id", "execution"} or any(
            type(record.get(k)) is not int or record[k] != value
            for k, value in (("seed", seed), ("task_id", task_id))
        ):
            return False
        validate_gp_execution(record["execution"], "gp-root-dag-v1")
        raw = record["recipe"]
        source = raw["source"]
        if any(
            type(source.get(k)) is not int or source[k] != value
            for k, value in (("seed", seed), ("task_id", task_id))
        ):
            return False
        if set(controls) != {"topology", "request", "base", "bindings"}:
            return False
        if not gp_payload_equal(raw["limits"], asdict(limits)):
            return False
        request = gp_column_request_from_dict(controls["request"], limits=limits)
        if not gp_payload_equal(asdict(request), source["request"]):
            return False
        if not gp_payload_equal(controls["bindings"], raw["bindings"]):
            return False
        if not _submitted_fields(controls["base"], raw["base"], limits):
            return False
        _, candidate, _, _ = preflight_gp_root_dag(raw, limits=limits)
        expected = recipe_from_dict(
            authored_semantic_preview(recursive_authored_preview(controls["base"])),
            limits=limits,
        )
        return gp_payload_equal(asdict(expected), asdict(candidate))
    except (KeyError, TypeError, AttributeError, IndexError, ValueError, OverflowError):
        return False
