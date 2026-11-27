"""Strict bounded component archives for source observation construction receipts."""

import json
from dataclasses import asdict, fields
from typing import Any, cast

from r0.modules.priors.lib.causal_limits import CausalLimits
from r0.modules.priors.lib.fitted_fit_decoding import decode_fitted_raw_draw
from r0.modules.priors.lib.fitted_tree_decoding import fitted_array
from r0.modules.priors.lib.function_metadata import admit_function_metadata
from r0.modules.priors.lib.generator_snapshot import GeneratorSnapshot
from r0.modules.priors.lib.network_innovation_recipe import NetworkInnovationRecipe
from r0.modules.priors.lib.network_observation_request import (
    NetworkObservationRequest,
    observation_draw_specs,
    observation_widths,
)
from r0.modules.priors.lib.network_observation_sampling import (
    NetworkObservationChannel,
    SampledNetworkObservations,
)
from r0.modules.priors.lib.network_recipe import NetworkRecipe
from r0.modules.priors.lib.raw_draw_records import RawDrawLimits, RawDrawRecord

_DEFAULT_LIMITS = CausalLimits()


def _object(data: object, kind: Any) -> dict[str, Any]:
    if type(data) is not dict or set(cast(dict[str, Any], data)) != {f.name for f in fields(kind)}:
        raise ValueError("network observation record fields differ")
    return dict(cast(dict[str, Any], data))


def network_observation_request_from_dict(
    data: object, *, limits: CausalLimits = _DEFAULT_LIMITS
) -> NetworkObservationRequest:
    from r0.modules.priors.lib.generate import recipe_from_dict

    admit_function_metadata(data, limits)
    try:
        values = _object(data, NetworkObservationRequest)
        source = recipe_from_dict(values["source"], limits=limits)
        if type(source) not in (NetworkRecipe, NetworkInnovationRecipe):
            raise ValueError("observation source requires a standalone network")
        values["source"] = source
        values["raw_limits"] = RawDrawLimits(**_object(values["raw_limits"], RawDrawLimits))
        result = NetworkObservationRequest(**values)
        if json.dumps(data, sort_keys=True, allow_nan=False) != json.dumps(
            asdict(result), sort_keys=True, allow_nan=False
        ):
            raise ValueError("observation request archive must retain complete source fields")
        return result
    except (TypeError, KeyError, AttributeError, OverflowError, RecursionError) as error:
        raise ValueError("malformed network observation request") from error


def network_observations_from_dict(
    data: object, *, limits: CausalLimits = _DEFAULT_LIMITS
) -> SampledNetworkObservations:
    admit_function_metadata(data, limits)
    try:
        values = _object(data, SampledNetworkObservations)
        request = network_observation_request_from_dict(values["request"], limits=limits)
        values["request"] = request
        count = sum(observation_widths(request))
        values["catalog"] = tuple(
            NetworkObservationChannel(**_object(item, NetworkObservationChannel))
            for item in fitted_array(values["catalog"], count)
        )
        for name, maximum in (("features", request.features), ("targets", request.outputs)):
            values[name] = fitted_array(values[name], maximum)
        for name in ("initial", "final"):
            snapshot = _object(values[name], GeneratorSnapshot)
            if (
                type(snapshot["payload"]) is not str
                or len(snapshot["payload"]) > request.raw_limits.state_bytes
            ):
                raise ValueError("observation snapshot exceeds declared byte cap")
            values[name] = GeneratorSnapshot(**snapshot)
        specs = observation_draw_specs(request)
        raw = fitted_array(values["raw"], len(specs))
        if len(raw) != len(specs):
            raise ValueError("observation raw count differs")
        records: list[RawDrawRecord] = []
        for item, spec in zip(raw, specs, strict=True):
            fields_raw = _object(item, RawDrawRecord)
            fitted_array(fields_raw["values"], spec.numbers)
            records.append(decode_fitted_raw_draw(fields_raw))
        values["raw"] = tuple(records)
        result = SampledNetworkObservations(**values)
        if json.dumps(data, sort_keys=True, allow_nan=False) != json.dumps(
            asdict(result), sort_keys=True, allow_nan=False
        ):
            raise ValueError("observation archive differs from exact replay types")
        return result
    except (TypeError, KeyError, AttributeError, OverflowError, RecursionError) as error:
        raise ValueError("malformed network observation envelope") from error
