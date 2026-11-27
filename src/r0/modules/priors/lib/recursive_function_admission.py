"""Aggregate sampled replay reservations before decoding any forest construction stream."""

from dataclasses import fields
from typing import Any, cast

from r0.modules.priors.lib.causal_limits import CausalLimits
from r0.modules.priors.lib.raw_draw_records import RawDrawLimits


def admit_recursive_payloads(
    envelopes: list[dict[str, Any]],
    *,
    storage: int,
    numbers: int,
    limits: CausalLimits,
    graphs: list[dict[str, Any]],
    convolutions: list[dict[str, Any]],
) -> None:
    from r0.modules.priors.lib.recursive_forest_decoding import recursive_forest_plan_from_dict
    from r0.modules.priors.lib.recursive_forest_plan import recursive_effective_limits

    work = 4 * storage
    metadata_numbers = numbers
    for envelope in envelopes:
        source = envelope.get("source")
        if type(source) is not dict:
            raise ValueError("recursive structural source must be an object")
        source = cast(dict[str, Any], source)
        if source.get("version") != "sampled-recursive-forest-v1":
            continue
        try:
            declared = source["limits"]
            raw = source["raw_limits"]
            if type(declared) is not dict or set(cast(dict[str, Any], declared)) != {
                f.name for f in fields(CausalLimits)
            }:
                raise ValueError("recursive sampled limits require complete fields")
            if type(raw) is not dict or set(cast(dict[str, Any], raw)) != {
                f.name for f in fields(RawDrawLimits)
            }:
                raise ValueError("recursive raw limits require complete fields")
            effective = recursive_effective_limits(
                CausalLimits(**cast(dict[str, Any], declared)), limits
            )
            plan = recursive_forest_plan_from_dict(source["plan"])
            if plan.lifecycle != "frozen-function":
                raise ValueError("per-row reference is not a frozen structural function")
            if plan.task == "classification" and plan.classes > effective.node_channels:
                raise ValueError("recursive indicator class width exceeds caller channel limit")
            report = plan.admit(effective, RawDrawLimits(**cast(dict[str, Any], raw)))
            work += report["estimated_working_bytes"]
            numbers += report["parameters"]
        except (KeyError, TypeError, AttributeError, OverflowError) as error:
            raise ValueError("malformed recursive sampled construction budget") from error
    if convolutions:
        from r0.modules.priors.lib.feature_conv_decoding import feature_conv1d_request_from_dict

        for convolution in convolutions:
            request = feature_conv1d_request_from_dict(convolution.get("request"), limits=limits)
            report = request.admit(limits=limits)
            work += report["estimated_working_bytes"]
            numbers += report["parameters"]
    if graphs:
        from r0.modules.priors.lib.recursive_task_preflight import (
            RECURSIVE_TASK_VERSIONS,
            admit_recursive_tasks,
        )

        admit_recursive_tasks(
            [g for g in graphs if g.get("version") in RECURSIVE_TASK_VERSIONS],
            storage=storage,
            numbers=numbers + metadata_numbers,
            replay_work=work - 4 * storage,
            limits=limits,
        )
    if work > limits.working_bytes or numbers > limits.parameters:
        raise ValueError("aggregate recursive archive replay workspace or parameter limit exceeded")
