"""Admit complete root provenance, then reopen by exact deterministic construction."""

from typing import Any, cast

from r0.modules.priors.lib.causal_limits import CausalLimits
from r0.modules.priors.lib.fitted_tree_decoding import fitted_array, fitted_object
from r0.modules.priors.lib.function_metadata import admit_function_metadata
from r0.modules.priors.lib.generator_snapshot import GeneratorSnapshot
from r0.modules.priors.lib.raw_draw_records import RawDrawLimits
from r0.modules.priors.lib.sampled_source_root_admission import admit_sampled_source_root
from r0.modules.priors.lib.sampled_source_root_base import (
    SampledSourceRootBase,
    sample_source_root_base,
)
from r0.modules.priors.lib.sampled_source_root_profile import SampledSourceRootProfile
from r0.modules.priors.lib.source_root_laws import source_root_from_dict
from r0.modules.priors.lib.source_root_profile_decoding import (
    sampled_source_root_profile_from_dict,
    source_root_profile_from_dict,
)

_DEFAULT_LIMITS = CausalLimits()


def sampled_source_root_base_from_dict(
    data: dict[str, Any],
    *,
    limits: CausalLimits = _DEFAULT_LIMITS,
) -> SampledSourceRootBase:
    try:
        return _decode(data, limits)
    except (TypeError, KeyError, AttributeError, OverflowError, RecursionError) as error:
        raise ValueError("malformed sampled source root archive") from error


def _decode(data: dict[str, Any], limits: CausalLimits) -> SampledSourceRootBase:
    from r0.modules.priors.lib.sampled_graph_decoding import construction_matches

    if type(limits) is not CausalLimits:
        raise ValueError("sampled root decoder requires causal limits")
    admit_function_metadata(data, limits)
    record = fitted_object(data, SampledSourceRootBase)
    declared = CausalLimits(**fitted_object(record["limits"], CausalLimits))
    effective = CausalLimits(
        **{
            key: min(getattr(declared, key), getattr(limits, key))
            for key in ("nodes", "node_channels", "total_channels", "parameters", "working_bytes")
        }
    )
    raw_limits = RawDrawLimits(**fitted_object(record["raw_limits"], RawDrawLimits))
    profile_record = fitted_object(record["profile"], SampledSourceRootProfile)
    request = source_root_profile_from_dict(profile_record["profile"])
    # Bound the law before constructing tuples and before profile snapshot replay.
    law_record = record["law"]
    if type(law_record) is not dict:
        raise ValueError("sampled root law must be an object")
    law_record = cast(dict[str, Any], law_record)
    coordinates = fitted_array(law_record.get("coordinates"), request.width)
    if len(coordinates) != request.width:
        raise ValueError("sampled root law width differs from profile")
    for coordinate in coordinates:
        if type(coordinate) is not dict:
            raise ValueError("sampled root coordinate must be an object")
        fitted_array(cast(dict[str, Any], coordinate).get("probabilities"), 10)
    law = source_root_from_dict(law_record)
    modes = fitted_array(record["modes"], request.width)
    rows = record["calibration_rows"]
    admit_sampled_source_root(request, law, modes, rows, effective, raw_limits)
    profile = sampled_source_root_profile_from_dict(record["profile"], limits=effective)
    if profile.law != law:
        raise ValueError("sampled root law differs from retained profile")
    rng = None
    if rows:
        initial = fitted_object(record["calibration_initial"], GeneratorSnapshot)
        rng = GeneratorSnapshot(**initial).restore(byte_limit=raw_limits.state_bytes)
    result = sample_source_root_base(
        profile,
        modes=modes,
        calibration_rows=rows,
        calibration_rng=rng,
        limits=declared,
        raw_limits=raw_limits,
    )
    if not construction_matches(result, data):
        raise ValueError("sampled source root archive differs from complete replay")
    return result
