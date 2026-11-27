"""Bounded strict source-root profile decoding before raw replay."""

from typing import Any, cast

from r0.modules.priors.lib.causal_limits import CausalLimits
from r0.modules.priors.lib.fitted_fit_decoding import decode_fitted_raw_draw
from r0.modules.priors.lib.fitted_tree_decoding import fitted_array, fitted_object
from r0.modules.priors.lib.generator_snapshot import GeneratorSnapshot
from r0.modules.priors.lib.raw_draw_records import RawDrawLimits
from r0.modules.priors.lib.sampled_source_root_profile import SampledSourceRootProfile
from r0.modules.priors.lib.source_root_laws import source_root_from_dict
from r0.modules.priors.lib.source_root_profiles import SourceRootProfile

_DEFAULT_LIMITS = CausalLimits()


def source_root_profile_from_dict(data: dict[str, Any]) -> SourceRootProfile:
    fields = fitted_object(data, SourceRootProfile)
    fields["families"] = fitted_array(fields["families"], 1024)
    fields["raw_limits"] = RawDrawLimits(**fitted_object(fields["raw_limits"], RawDrawLimits))
    return SourceRootProfile(**fields)


def sampled_source_root_profile_from_dict(
    data: dict[str, Any],
    *,
    limits: CausalLimits = _DEFAULT_LIMITS,
) -> SampledSourceRootProfile:
    fields = fitted_object(data, SampledSourceRootProfile)
    profile = source_root_profile_from_dict(fields["profile"])
    profile.admit(limits)
    fields["profile"] = profile
    fields["limits"] = CausalLimits(**fitted_object(fields["limits"], CausalLimits))
    law = fields["law"]
    if (
        type(law) is not dict
        or len(fitted_array(cast(dict[str, Any], law).get("coordinates"), profile.width))
        != profile.width
    ):
        raise ValueError("root law geometry differs from profile")
    for coordinate in cast(dict[str, Any], law)["coordinates"]:
        if type(coordinate) is not dict:
            raise ValueError("root coordinate must be a JSON object")
        fitted_array(cast(dict[str, Any], coordinate).get("probabilities"), 10)
    fields["law"] = source_root_from_dict(cast(dict[str, Any], law))
    gaussian = fitted_array(fields["model_gaussians"], profile.gaussian_width)
    for coordinate in gaussian:
        if (
            type(coordinate) is not dict
            or cast(dict[str, Any], coordinate).get("family") != "gaussian"
        ):
            raise ValueError("model root parameters must be Gaussian")
        fitted_array(cast(dict[str, Any], coordinate).get("probabilities"), 0)
    if len(gaussian) != profile.gaussian_width:
        raise ValueError("root Gaussian metadata geometry differs")
    fields["model_gaussians"] = (
        source_root_from_dict(
            {"coordinates": gaussian, "version": "source-root-law-v1"}
        ).coordinates
        if gaussian
        else ()
    )
    fields["thresholds"] = fitted_array(fields["thresholds"], 3)
    specs = profile.worst_draws()
    raw = fitted_array(fields["raw"], len(specs))
    max_values = max((s.numbers for s in specs), default=0)
    for record in raw:
        if (
            type(record) is not dict
            or len(fitted_array(cast(dict[str, Any], record).get("values"), max_values))
            > max_values
        ):
            raise ValueError("root raw record exceeds profile bound")
        obj = cast(dict[str, Any], record)
        fitted_array(obj.get("owner"), 2)
        fitted_array(obj.get("shape"), 1)
        for pair in fitted_array(obj.get("parameters"), 2):
            values = fitted_array(pair, 2)
            if len(values) != 2:
                raise ValueError("root raw parameter pair differs")
            if type(values[1]) in (list, tuple):
                fitted_array(values[1], 10)
    if sum(len(record["values"]) for record in raw) > sum(s.numbers for s in specs):
        raise ValueError("root raw record total exceeds profile bound")
    fields["raw"] = tuple(decode_fitted_raw_draw(record) for record in raw)
    for key in ("initial_rng", "final_rng"):
        snapshots: list[GeneratorSnapshot] = []
        for record in fitted_array(fields[key], 2):
            obj = fitted_object(record, GeneratorSnapshot)
            if (
                type(obj["payload"]) is not str
                or len(obj["payload"]) > profile.raw_limits.state_bytes
            ):
                raise ValueError("root snapshot exceeds declared bound")
            snapshots.append(GeneratorSnapshot(**obj))
        fields[key] = tuple(snapshots)
    return SampledSourceRootProfile(**fields)
