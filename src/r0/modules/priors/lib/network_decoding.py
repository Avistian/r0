"""Strict bounded neural architecture decoding and complete sampling replay."""

from dataclasses import fields as dataclass_fields
from typing import Any, cast

from r0.modules.priors.lib.causal_activations import CausalActivation
from r0.modules.priors.lib.causal_limits import CausalLimits
from r0.modules.priors.lib.function_metadata import admit_function_metadata
from r0.modules.priors.lib.structural_network import (
    FrozenNetworkFunction,
    NetworkAffine,
    NetworkBlock,
    NetworkProjection,
)


def record(data: Any, cls: type) -> dict[str, Any]:
    if type(data) is not dict or set(cast(dict[str, Any], data)) != {
        f.name for f in dataclass_fields(cls)
    }:
        raise ValueError("network records require every declared field and no extras")
    return dict(cast(dict[str, Any], data))


def array(data: Any) -> tuple[Any, ...]:
    if type(data) not in (tuple, list):
        raise ValueError("network arrays must be lists or tuples")
    return tuple(data)


def activation(data: Any) -> CausalActivation:
    fields = record(data, CausalActivation)
    fields["parameters"] = tuple(array(p) for p in array(fields["parameters"]))
    return CausalActivation(**fields)


def affine(data: Any) -> NetworkAffine:
    fields = record(data, NetworkAffine)
    for name in ("matrix", "mask"):
        fields[name] = tuple(array(row) for row in array(fields[name]))
    fields["bias"] = array(fields["bias"])
    fields["activation"] = activation(fields["activation"])
    return NetworkAffine(**fields)


def decode_network(data: Any) -> FrozenNetworkFunction:
    fields = record(data, FrozenNetworkFunction)
    blocks: list[NetworkBlock] = []
    for item in array(fields["blocks"]):
        b = record(item, NetworkBlock)
        b["parents"] = array(b["parents"])
        b["experts"] = tuple(affine(e) for e in array(b["experts"]))
        b["gate"] = affine(b["gate"]) if b["gate"] is not None else None
        skips: list[NetworkProjection] = []
        for item in array(b["skips"]):
            s = record(item, NetworkProjection)
            s["affine"] = affine(s["affine"])
            skips.append(NetworkProjection(**s))
        b["skips"] = tuple(skips)
        blocks.append(NetworkBlock(**b))
    fields["blocks"] = tuple(blocks)
    return FrozenNetworkFunction(**fields)


def decode_sampled_network(data: Any, limits: CausalLimits):
    from r0.modules.priors.lib.generator_snapshot import GeneratorSnapshot
    from r0.modules.priors.lib.network_sampling import sample_network
    from r0.modules.priors.lib.network_sampling_records import (
        SampledNetwork,
    )
    from r0.modules.priors.lib.sampled_graph_decoding import construction_matches

    admit_function_metadata(data, limits)
    fields = record(data, SampledNetwork)
    request = decode_network_request(fields["request"], limits)
    initial = array(fields["initial"])
    if len(initial) != 2:
        raise ValueError("network requires two initial streams")
    streams = [
        GeneratorSnapshot(**record(s, GeneratorSnapshot)).restore(
            byte_limit=request.raw_limits.state_bytes
        )
        for s in initial
    ]
    result = sample_network(request, named_rng=streams[0], raw_rng=streams[1])
    if not construction_matches(result, data):
        raise ValueError("network archive differs from complete replay")
    return result


def decode_network_request(data: Any, limits: CausalLimits):
    from r0.modules.priors.lib.network_sampling_records import NetworkSamplingRequest
    from r0.modules.priors.lib.raw_draw_records import RawDrawLimits

    req = record(data, NetworkSamplingRequest)
    declared = CausalLimits(**record(req["limits"], CausalLimits))
    if any(getattr(declared, f.name) > getattr(limits, f.name) for f in dataclass_fields(limits)):
        raise ValueError("network declared limits exceed caller limits")
    req["limits"] = declared
    req["raw_limits"] = RawDrawLimits(**record(req["raw_limits"], RawDrawLimits))
    for name in ("widths", "families", "weight_laws"):
        req[name] = array(req[name])
    req["activations"] = tuple(activation(a) for a in array(req["activations"]))
    return NetworkSamplingRequest(**req)
