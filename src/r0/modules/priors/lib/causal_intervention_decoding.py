"""Bounded explicit intervention archive decoding without refitting mechanisms."""

from typing import Any, cast

from r0.modules.priors.lib.causal_intervention_records import (
    CausalInterventionRecipe,
    HardIntervention,
    SoftIntervention,
)
from r0.modules.priors.lib.causal_limits import CausalLimits
from r0.modules.priors.lib.converted_decoding import categorical_converter_from_dict
from r0.modules.priors.lib.converter_intervention import (
    ConverterHardIntervention,
    ConverterSoftIntervention,
)
from r0.modules.priors.lib.exogenous import ExogenousDistribution, NoiseResponse
from r0.modules.priors.lib.frozen_function_decoding import frozen_function_from_dict
from r0.modules.priors.lib.function_metadata import admit_function_metadata
from r0.modules.priors.lib.functional_causal_decoding import (
    functional_array,
    functional_pipeline,
    functional_record,
    functional_root,
)
from r0.modules.priors.lib.inputs import InputDistribution
from r0.modules.priors.lib.native_causal import FrequencyVariableDomain, VariableDomain
from r0.modules.priors.lib.native_vector_intervention import (
    NativeVectorDomainIntervention,
    NativeVectorHardIntervention,
)
from r0.modules.priors.lib.stage_intervention import StageIntervention


def intervention_from_dict(data: object, *, limits: CausalLimits) -> CausalInterventionRecipe:
    from r0.modules.priors.lib.generate import recipe_from_dict

    admit_function_metadata(data, limits)
    try:
        raw = functional_record(data, CausalInterventionRecipe)
        base = recipe_from_dict(raw["base"], limits=limits)
        changes: list[
            HardIntervention
            | SoftIntervention
            | StageIntervention
            | NativeVectorHardIntervention
            | NativeVectorDomainIntervention
            | ConverterHardIntervention
            | ConverterSoftIntervention
        ] = []
        for item in functional_array(raw["interventions"]):
            if type(item) is not dict:
                raise ValueError("intervention must be a versioned object")
            if cast(dict[str, Any], item).get("version") == "converter-hard-intervention-v1":
                changes.append(
                    ConverterHardIntervention(**functional_record(item, ConverterHardIntervention))
                )
            elif cast(dict[str, Any], item).get("version") == "converter-soft-intervention-v1":
                change = functional_record(item, ConverterSoftIntervention)
                change["converter"] = categorical_converter_from_dict(
                    change["converter"], limits=limits
                )
                changes.append(ConverterSoftIntervention(**change))
            elif cast(dict[str, Any], item).get("version") == "native-vector-hard-intervention-v1":
                changes.append(
                    NativeVectorHardIntervention(
                        **functional_record(item, NativeVectorHardIntervention)
                    )
                )
            elif (
                cast(dict[str, Any], item).get("version") == "native-vector-domain-intervention-v1"
            ):
                change = functional_record(item, NativeVectorDomainIntervention)
                raw_domain = change["domain"]
                domain_type = (
                    FrequencyVariableDomain
                    if isinstance(raw_domain, dict)
                    and cast(dict[str, Any], raw_domain).get("version")
                    == "native-frequency-domain-v2"
                    else VariableDomain
                )
                domain = functional_record(raw_domain, domain_type)
                for key in ("vocabulary", "thresholds", "frequencies"):
                    if key in domain:
                        domain[key] = functional_array(domain[key])
                change["domain"] = domain_type(**domain)
                changes.append(NativeVectorDomainIntervention(**change))
            elif cast(dict[str, Any], item).get("version") == "hard-causal-intervention-v1":
                change = functional_record(item, HardIntervention)
                for key in ("channels", "values"):
                    change[key] = functional_array(change[key])
                changes.append(HardIntervention(**change))
            elif cast(dict[str, Any], item).get("version") == "stage-causal-intervention-v1":
                change = functional_record(item, StageIntervention)
                if change.get("root") is not None:
                    change["root"] = functional_root(change["root"])
                if change.get("pipeline") is not None:
                    change["pipeline"] = functional_pipeline(change["pipeline"])
                if change.get("domain") is not None:
                    raw_domain = change["domain"]
                    if type(raw_domain) is not dict:
                        raise ValueError("stage domain must be an object")
                    domain_type = (
                        FrequencyVariableDomain
                        if cast(dict[str, Any], raw_domain).get("version")
                        == "native-frequency-domain-v2"
                        else VariableDomain
                    )
                    domain = functional_record(raw_domain, domain_type)
                    for key in ("vocabulary", "thresholds", "frequencies"):
                        if key in domain:
                            domain[key] = functional_array(domain[key])
                    change["domain"] = domain_type(**domain)
                changes.append(StageIntervention(**change))
            else:
                change = functional_record(item, SoftIntervention)
                if change["function"] is not None:
                    change["function"] = frozen_function_from_dict(
                        change["function"], limits=limits
                    )
                if change["noise_scale"] is not None:
                    change["noise_scale"] = functional_array(change["noise_scale"])
                if change["noise_response"] is not None:
                    change["noise_response"] = NoiseResponse(
                        **functional_record(change["noise_response"], NoiseResponse)
                    )
                if change["exogenous"] is not None:
                    noise = functional_record(change["exogenous"], ExogenousDistribution)
                    noise["inputs"] = InputDistribution(
                        **functional_record(noise["inputs"], InputDistribution)
                    )
                    noise["factor"] = tuple(
                        functional_array(row) for row in functional_array(noise["factor"])
                    )
                    change["exogenous"] = ExogenousDistribution(**noise)
                changes.append(SoftIntervention(**change))
        raw.update(base=base, interventions=tuple(changes))
        return CausalInterventionRecipe(**raw)
    except (TypeError, KeyError, OverflowError, RecursionError) as error:
        raise ValueError("malformed causal intervention archive") from error
