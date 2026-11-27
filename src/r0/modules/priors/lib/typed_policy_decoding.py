"""Strict typed receipts with metadata-only admission before nested snapshot replay."""

from dataclasses import fields, replace
from typing import Any, cast

from r0.modules.priors.lib.causal_limits import CausalLimits
from r0.modules.priors.lib.function_metadata import admit_function_metadata
from r0.modules.priors.lib.generator_snapshot import GeneratorSnapshot
from r0.modules.priors.lib.intervention_policy_decoding import complete_record
from r0.modules.priors.lib.raw_draw_capture import RawDrawLedger
from r0.modules.priors.lib.raw_draw_records import RawDrawLimits
from r0.modules.priors.lib.raw_draw_spec import RawDrawSpec
from r0.modules.priors.lib.sampled_graph_decoding import construction_matches
from r0.modules.priors.lib.typed_policy_records import SampledTypedInterventionPolicy
from r0.modules.priors.lib.typed_policy_request import TypedInterventionPolicyRequest


def stored_base_limits(raw: dict[str, Any]) -> CausalLimits:
    declared = raw["limits"] if "limits" in raw else raw["request"]["limits"]
    return CausalLimits(**complete_record(declared, CausalLimits))


def restore_typed_base(raw: dict[str, Any]):
    from r0.modules.priors.lib.generate import recipe_from_dict

    return recipe_from_dict(raw, limits=stored_base_limits(raw))


def typed_policy_from_dict(data: Any, *, limits: CausalLimits) -> SampledTypedInterventionPolicy:
    from r0.modules.priors.lib.typed_policy_request_decoding import (
        typed_intervention_policy_request_from_dict,
    )
    from r0.modules.priors.lib.typed_policy_sampling import sample_typed_intervention_policy

    admit_function_metadata(data, limits)
    try:
        raw = complete_record(data, SampledTypedInterventionPolicy)
        request = complete_record(raw["request"], TypedInterventionPolicyRequest)
        declared = CausalLimits(**complete_record(request["limits"], CausalLimits))
        effective = CausalLimits(
            **{
                f.name: min(getattr(declared, f.name), getattr(limits, f.name))
                for f in fields(CausalLimits)
            }
        )
        raw_limits = RawDrawLimits(**complete_record(request["raw_limits"], RawDrawLimits))
        bounded_raw = replace(
            raw_limits, working_bytes=min(raw_limits.working_bytes, effective.working_bytes)
        )
        if any(type(raw[group]) not in (list, tuple) for group in ("raw", "reference_raw")):
            raise ValueError("typed raw custody requires arrays")
        count = len(raw["raw"]) + len(raw["reference_raw"])
        if (
            count > raw_limits.draws
            or 128 * count + 6 * raw_limits.state_bytes > bounded_raw.working_bytes
        ):
            raise ValueError("typed raw/snapshot metadata reserve exceeds limits")
        specs: list[RawDrawSpec] = []
        retained = 0

        def freeze(v: Any) -> Any:
            return (
                tuple(freeze(x) for x in cast(list[Any] | tuple[Any, ...], v))
                if type(v) in (list, tuple)
                else v
            )

        for group in ("raw", "reference_raw"):
            if type(raw[group]) not in (list, tuple):
                raise ValueError("typed raw custody requires arrays")
            for item in raw[group]:
                specs.append(
                    RawDrawSpec(
                        freeze(item["owner"]),
                        item["role"],
                        item["distribution"],
                        freeze(item["shape"]),
                        freeze(item["parameters"]),
                    )
                )
                retained += specs[-1].retained_bytes(item["stream"])
        RawDrawLedger(limits=bounded_raw).admit_specs(tuple(specs))
        # These outer costs are already knowable from metadata. Reserve them with
        # reference geometry before decoding any sampled base or restoring its RNGs.
        reserved = 6 * raw_limits.state_bytes + retained
        if reserved > bounded_raw.working_bytes:
            raise ValueError("typed policy raw/snapshot reservation exceeds caller limits")
        typed = typed_intervention_policy_request_from_dict(request, limits=effective)
        if type(raw["initial"]) not in (list, tuple) or len(raw["initial"]) != 3:
            raise ValueError("typed policy requires three initial snapshots")
        streams = [
            GeneratorSnapshot(**complete_record(s, GeneratorSnapshot)).restore(
                byte_limit=raw_limits.state_bytes
            )
            for s in raw["initial"]
        ]
        result = sample_typed_intervention_policy(
            typed, selection_rng=streams[0], value_rng=streams[1], calibration_rng=streams[2]
        )
        if not construction_matches(result, data):
            raise ValueError("typed policy archive differs from complete exact replay")
        return result
    except (KeyError, TypeError, AttributeError, IndexError, OverflowError) as error:
        raise ValueError("malformed typed intervention policy archive") from error
