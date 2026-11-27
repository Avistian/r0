"""Strict observation task decoding with aggregate admission before source replay."""

from dataclasses import fields
from typing import Any

from r0.modules.priors.lib.causal_limits import CausalLimits
from r0.modules.priors.lib.generate import recipe_from_dict
from r0.modules.priors.lib.linear import LinearRecipe
from r0.modules.priors.lib.network_decoding import record
from r0.modules.priors.lib.network_observation_decoding import network_observations_from_dict
from r0.modules.priors.lib.network_observation_recipe import NetworkObservationRecipe
from r0.modules.priors.lib.observation_envelope_admission import (
    admit_observation_envelope,
    observation_metadata_cost,
)
from r0.modules.priors.lib.observation_source_preview import field
from r0.modules.priors.lib.sampled_graph_decoding import construction_matches


def network_observation_recipe_from_dict(
    data: Any, *, limits: CausalLimits
) -> NetworkObservationRecipe:
    if type(limits) is not CausalLimits:
        raise ValueError("observation decoder requires caller-owned limits")
    observation_metadata_cost(data, limits)
    try:
        raw = record(data, NetworkObservationRecipe)
        task = recipe_from_dict(raw["task"], limits=limits)
        if type(task) is not LinearRecipe:
            raise ValueError("observation task controls differ")
        source = field(field(raw["observations"], "request"), "source")
        declared = CausalLimits(**record(field(source, "limits"), CausalLimits))
        if any(getattr(declared, f.name) > getattr(limits, f.name) for f in fields(CausalLimits)):
            raise ValueError("observation declared limits exceed caller limits")
        admit_observation_envelope(task, raw["observations"], declared)
        receipt = network_observations_from_dict(raw["observations"], limits=declared)
        result = NetworkObservationRecipe(task, receipt, raw["version"])
        if not construction_matches(result, data):
            raise ValueError("observation recipe differs from complete exact source replay")
        return result
    except (KeyError, TypeError, AttributeError, OverflowError, RecursionError) as error:
        raise ValueError("malformed observation task envelope") from error
