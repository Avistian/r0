"""Strict profile request decoding and replay of both scalar and network custody."""

from typing import Any

from r0.modules.priors.lib.causal_limits import CausalLimits
from r0.modules.priors.lib.function_metadata import admit_function_metadata
from r0.modules.priors.lib.generator_snapshot import GeneratorSnapshot
from r0.modules.priors.lib.network_decoding import array, decode_network_request, record
from r0.modules.priors.lib.network_hyperprior_records import (
    NetworkHyperpriorRequest,
    SampledHyperpriorNetwork,
)
from r0.modules.priors.lib.network_hyperprior_sampling import sample_hyperprior_network
from r0.modules.priors.lib.network_scalar_priors import NetworkScalarPrior
from r0.modules.priors.lib.sampled_graph_decoding import construction_matches


def hyperprior_request_from_dict(data: Any, *, limits: CausalLimits) -> NetworkHyperpriorRequest:
    admit_function_metadata(data, limits)
    values = record(data, NetworkHyperpriorRequest)
    values["template"] = decode_network_request(values["template"], limits)
    for name in ("depth", "width", "weight_scale", "bias_scale"):
        if values[name] is None and name == "bias_scale":
            continue
        prior = record(values[name], NetworkScalarPrior)
        prior["choices"] = array(prior["choices"])
        values[name] = NetworkScalarPrior(**prior)
    return NetworkHyperpriorRequest(**values)


def decode_hyperprior_network(data: Any, limits: CausalLimits) -> SampledHyperpriorNetwork:
    admit_function_metadata(data, limits)
    values = record(data, SampledHyperpriorNetwork)
    request = hyperprior_request_from_dict(values["profile"], limits=limits)
    initial = array(values["initial"])
    if len(initial) != 2:
        raise ValueError("profiled network requires two child streams")
    streams = [
        GeneratorSnapshot(**record(s, GeneratorSnapshot)).restore(
            byte_limit=request.template.raw_limits.state_bytes
        )
        for s in (values["hyper_initial"], *initial)
    ]
    result = sample_hyperprior_network(
        request, hyper_rng=streams[0], named_rng=streams[1], raw_rng=streams[2]
    )
    if not construction_matches(result, data):
        raise ValueError("profiled network archive differs from full scalar/network replay")
    return result
