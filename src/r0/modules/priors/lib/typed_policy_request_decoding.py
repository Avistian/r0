"""Strict request admission and original envelope reconstruction before policy draws."""

from dataclasses import fields, replace
from typing import Any

from r0.modules.priors.lib.causal_limits import CausalLimits
from r0.modules.priors.lib.function_metadata import admit_function_metadata
from r0.modules.priors.lib.intervention_policy_decoding import complete_record
from r0.modules.priors.lib.intervention_policy_laws import InterventionParameterLaw
from r0.modules.priors.lib.raw_draw_records import RawDrawLimits
from r0.modules.priors.lib.typed_policy_locations import InterventionPolicyLocation
from r0.modules.priors.lib.typed_policy_request import TypedInterventionPolicyRequest

_DEFAULT_LIMITS = CausalLimits()


def typed_intervention_policy_request_from_dict(
    data: Any, *, limits: CausalLimits = _DEFAULT_LIMITS
) -> TypedInterventionPolicyRequest:
    from r0.modules.priors.lib.generate import recipe_from_dict
    from r0.modules.priors.lib.typed_policy_admission import admit_typed_policy
    from r0.modules.priors.lib.typed_policy_decoding import restore_typed_base
    from r0.modules.priors.lib.typed_policy_planning import admit_typed_record

    admit_function_metadata(data, limits)
    try:
        request = complete_record(data, TypedInterventionPolicyRequest)
        declared = CausalLimits(**complete_record(request["limits"], CausalLimits))
        effective = CausalLimits(
            **{
                f.name: min(getattr(declared, f.name), getattr(limits, f.name))
                for f in fields(limits)
            }
        )
        raw_limits = RawDrawLimits(**complete_record(request["raw_limits"], RawDrawLimits))
        admit_typed_record(request, effective)
        recipe_from_dict(request["base"], limits=effective)
        request["base"] = restore_typed_base(request["base"])
        request["limits"], request["raw_limits"] = declared, raw_limits
        for key in ("label_nodes", "eligibility", "locations", "parameter_laws"):
            if type(request[key]) not in (list, tuple):
                raise ValueError("typed policy fields require arrays")
            request[key] = tuple(request[key])
        locations: list[InterventionPolicyLocation] = []
        for item in request["locations"]:
            address = complete_record(item, InterventionPolicyLocation)
            for key in ("channels", "probabilities"):
                if type(address[key]) not in (list, tuple):
                    raise ValueError("typed location fields require arrays")
                address[key] = tuple(address[key])
            locations.append(InterventionPolicyLocation(**address))
        request["locations"] = tuple(locations)
        laws: list[InterventionParameterLaw] = []
        for item in request["parameter_laws"]:
            law = complete_record(item, InterventionParameterLaw)
            for key in ("path", "choices"):
                if type(law[key]) not in (list, tuple):
                    raise ValueError("typed law fields require arrays")
                law[key] = tuple(law[key])
            laws.append(InterventionParameterLaw(**law))
        request["parameter_laws"] = tuple(laws)
        typed = TypedInterventionPolicyRequest(**request)
        admit_typed_policy(replace(typed, limits=effective))
        return typed
    except (KeyError, TypeError, AttributeError, IndexError, OverflowError) as error:
        raise ValueError("malformed typed intervention policy request") from error
