"""Strict replay reconstruction of sampled policy envelopes, never loose normalization."""

from dataclasses import fields, replace
from typing import Any, cast

from r0.modules.priors.lib.causal_limits import CausalLimits
from r0.modules.priors.lib.function_metadata import admit_function_metadata
from r0.modules.priors.lib.generator_snapshot import GeneratorSnapshot
from r0.modules.priors.lib.intervention_policy_laws import InterventionParameterLaw
from r0.modules.priors.lib.intervention_policy_records import SampledInterventionPolicy
from r0.modules.priors.lib.intervention_policy_request import InterventionPolicyRequest
from r0.modules.priors.lib.intervention_policy_sampling import (
    admit_intervention_policy,
    sample_intervention_policy,
)
from r0.modules.priors.lib.raw_draw_capture import RawDrawLedger
from r0.modules.priors.lib.raw_draw_records import RawDrawLimits
from r0.modules.priors.lib.raw_draw_spec import RawDrawSpec
from r0.modules.priors.lib.sampled_graph_decoding import construction_matches


def complete_record(data: Any, kind: type[Any]) -> dict[str, Any]:
    if type(data) is not dict:
        raise ValueError("policy archive requires an object")
    mapping = cast(dict[str, Any], data)
    if set(mapping) != {f.name for f in fields(kind)}:
        raise ValueError("policy archive requires every field and no unknown fields")
    return dict(mapping)


def intervention_policy_from_dict(data: Any, *, limits: CausalLimits) -> SampledInterventionPolicy:
    from r0.modules.priors.lib.generate import recipe_from_dict

    admit_function_metadata(data, limits)
    try:
        raw = complete_record(data, SampledInterventionPolicy)
        r = complete_record(raw["request"], InterventionPolicyRequest)
        declared = CausalLimits(**complete_record(r["limits"], CausalLimits))
        effective = CausalLimits(
            **{
                f.name: min(getattr(limits, f.name), getattr(declared, f.name))
                for f in fields(CausalLimits)
            }
        )
        raw_limits = RawDrawLimits(**complete_record(r["raw_limits"], RawDrawLimits))
        specifications: list[RawDrawSpec] = []

        def freeze(value: Any) -> Any:
            return (
                tuple(freeze(v) for v in cast(list[Any] | tuple[Any, ...], value))
                if type(value) in (list, tuple)
                else value
            )

        for name in ("raw", "reference_raw"):
            if type(raw[name]) not in (list, tuple):
                raise ValueError("policy raw custody requires arrays")
            for draw in raw[name]:
                if type(draw) is not dict:
                    raise ValueError("policy raw custody requires complete draw records")
                draw = cast(dict[str, Any], draw)
                specifications.append(
                    RawDrawSpec(
                        freeze(draw["owner"]),
                        draw["role"],
                        draw["distribution"],
                        freeze(draw["shape"]),
                        freeze(draw["parameters"]),
                    )
                )
        RawDrawLedger(limits=raw_limits).admit_specs(tuple(specifications))
        recipe_from_dict(r["base"], limits=effective)
        base_limits = CausalLimits(**complete_record(r["base"]["limits"], CausalLimits))
        r["base"] = recipe_from_dict(r["base"], limits=base_limits)
        r["limits"] = declared
        r["raw_limits"] = RawDrawLimits(**complete_record(r["raw_limits"], RawDrawLimits))
        for name in ("label_nodes", "eligibility", "parameter_laws"):
            if type(r[name]) not in (tuple, list):
                raise ValueError("policy arrays require actual arrays")
            r[name] = tuple(r[name])
        laws: list[InterventionParameterLaw] = []
        for item in r["parameter_laws"]:
            law = complete_record(item, InterventionParameterLaw)
            for name in ("path", "choices"):
                if type(law[name]) not in (list, tuple):
                    raise ValueError("parameter law arrays require arrays")
                law[name] = tuple(law[name])
            laws.append(InterventionParameterLaw(**law))
        r["parameter_laws"] = tuple(laws)
        request = InterventionPolicyRequest(**r)
        admit_intervention_policy(replace(request, limits=effective))
        if type(raw["initial"]) not in (tuple, list) or len(raw["initial"]) != 3:
            raise ValueError("policy requires exactly three initial snapshots")
        streams = tuple(
            GeneratorSnapshot(**complete_record(s, GeneratorSnapshot)).restore(
                byte_limit=request.raw_limits.state_bytes
            )
            for s in raw["initial"]
        )
        result = sample_intervention_policy(
            request, selection_rng=streams[0], value_rng=streams[1], calibration_rng=streams[2]
        )
        if not construction_matches(result, data):
            raise ValueError("sampled policy archive differs from exact complete replay")
        return result
    except (KeyError, TypeError, AttributeError, IndexError, OverflowError) as error:
        raise ValueError("malformed sampled intervention policy archive") from error
