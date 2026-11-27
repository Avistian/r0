"""Bounded explicit intervention archive decoding without refitting mechanisms."""

from typing import Any, cast

from r0.modules.priors.lib.causal_intervention_records import (
    CausalInterventionRecipe,
    HardIntervention,
    SoftIntervention,
)
from r0.modules.priors.lib.causal_limits import CausalLimits
from r0.modules.priors.lib.exogenous import ExogenousDistribution, NoiseResponse
from r0.modules.priors.lib.frozen_function_decoding import frozen_function_from_dict
from r0.modules.priors.lib.function_metadata import admit_function_metadata
from r0.modules.priors.lib.functional_causal_decoding import functional_array, functional_record
from r0.modules.priors.lib.inputs import InputDistribution


def intervention_from_dict(data: object, *, limits: CausalLimits) -> CausalInterventionRecipe:
    from r0.modules.priors.lib.generate import recipe_from_dict

    admit_function_metadata(data, limits)
    try:
        raw = functional_record(data, CausalInterventionRecipe)
        base = recipe_from_dict(raw["base"], limits=limits)
        changes: list[HardIntervention | SoftIntervention] = []
        for item in functional_array(raw["interventions"]):
            if type(item) is not dict:
                raise ValueError("intervention must be a versioned object")
            if cast(dict[str, Any], item).get("version") == "hard-causal-intervention-v1":
                change = functional_record(item, HardIntervention)
                for key in ("channels", "values"):
                    change[key] = functional_array(change[key])
                changes.append(HardIntervention(**change))
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
