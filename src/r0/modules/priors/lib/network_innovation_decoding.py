"""Bounded source and compiled-equation decoding; no silent runtime repair."""

from dataclasses import fields as dataclass_fields
from typing import Any

from r0.modules.priors.lib.causal_limits import CausalLimits
from r0.modules.priors.lib.exogenous import ExogenousDistribution, NoiseResponse
from r0.modules.priors.lib.function_metadata import admit_function_metadata
from r0.modules.priors.lib.functional_causal_decoding import functional_node_from_dict
from r0.modules.priors.lib.inputs import InputDistribution
from r0.modules.priors.lib.network_decoding import array, record
from r0.modules.priors.lib.network_innovation_plan import (
    NetworkInnovationPlan,
    NetworkInnovationStage,
)
from r0.modules.priors.lib.network_innovation_recipe import NetworkInnovationRecipe
from r0.modules.priors.lib.network_recipe import NetworkRecipe, network_recipe_from_dict


def decode_innovation_plan(data: Any) -> NetworkInnovationPlan:
    fields = record(data, NetworkInnovationPlan)
    stages: list[NetworkInnovationStage | None] = []
    for item in array(fields["stages"]):
        if item is None:
            stages.append(None)
            continue
        stage = record(item, NetworkInnovationStage)
        stage["means"] = array(stage["means"])
        stage["scales"] = array(stage["scales"])
        exogenous = record(stage["exogenous"], ExogenousDistribution)
        exogenous["inputs"] = InputDistribution(**record(exogenous["inputs"], InputDistribution))
        exogenous["factor"] = tuple(array(row) for row in array(exogenous["factor"]))
        stage["exogenous"] = ExogenousDistribution(**exogenous)
        stage["response"] = NoiseResponse(**record(stage["response"], NoiseResponse))
        stages.append(NetworkInnovationStage(**stage))
    fields["stages"] = tuple(stages)
    fields["importance"] = array(fields["importance"])
    return NetworkInnovationPlan(**fields)


def network_innovation_from_dict(data: Any, *, limits: CausalLimits) -> NetworkInnovationRecipe:
    admit_function_metadata(data, limits)
    try:
        fields = record(data, NetworkInnovationRecipe)
        declared = CausalLimits(**record(fields["limits"], CausalLimits))
        if any(
            getattr(declared, f.name) > getattr(limits, f.name) for f in dataclass_fields(limits)
        ):
            raise ValueError("stochastic network declared limits exceed caller limits")
        fields["limits"] = declared
        fields["source"] = network_recipe_from_dict(fields["source"], limits=declared)
        from r0.modules.priors.lib.generate import recipe_from_dict

        fields["task"] = recipe_from_dict(fields["task"], limits=declared)
        fields["plan"] = decode_innovation_plan(fields["plan"])
        if fields["sampling"] is not None:
            fields["sampling"] = decode_sampled_innovations(fields["sampling"], fields["source"])
        fields["nodes"] = tuple(functional_node_from_dict(n) for n in array(fields["nodes"]))
        for name in ("observed", "target"):
            fields[name] = tuple(array(r) for r in array(fields[name]))
        return NetworkInnovationRecipe(**fields)
    except (KeyError, TypeError, AttributeError, OverflowError, RecursionError) as error:
        raise ValueError("malformed stochastic network archive") from error


def decode_sampled_innovations(data: Any, source: NetworkRecipe):
    from r0.modules.priors.lib.generator_snapshot import GeneratorSnapshot
    from r0.modules.priors.lib.network_innovation_sampling import sample_innovation_record
    from r0.modules.priors.lib.network_innovation_sampling_records import (
        NetworkInnovationRequest,
        SampledNetworkInnovations,
    )
    from r0.modules.priors.lib.network_scalar_priors import NetworkScalarPrior
    from r0.modules.priors.lib.raw_draw_records import RawDrawLimits
    from r0.modules.priors.lib.sampled_graph_decoding import construction_matches

    fields = record(data, SampledNetworkInnovations)
    request = record(fields["request"], NetworkInnovationRequest)
    for name in ("scale", "mean", "importance"):
        if request[name] is None:
            continue
        prior = record(request[name], NetworkScalarPrior)
        prior["choices"] = array(prior["choices"])
        request[name] = NetworkScalarPrior(**prior)
    request["raw_limits"] = RawDrawLimits(**record(request["raw_limits"], RawDrawLimits))
    specification = NetworkInnovationRequest(**request)
    initial = GeneratorSnapshot(**record(fields["initial"], GeneratorSnapshot))
    result = sample_innovation_record(
        source, specification, initial.restore(byte_limit=specification.raw_limits.state_bytes)
    )
    if not construction_matches(result, data):
        raise ValueError("network noise statistics archive differs from full construction replay")
    return result
