"""Whole retained template stream headers and raw shapes before the first replay."""

from typing import Any, cast

from r0.modules.priors.lib.causal_limits import CausalLimits
from r0.modules.priors.lib.feature_conv_decoding import (
    decode_feature_conv1d,
    feature_conv1d_request_from_dict,
)
from r0.modules.priors.lib.feature_conv_sampling_records import SampledFeatureConv1D
from r0.modules.priors.lib.generator_snapshot import GeneratorSnapshot
from r0.modules.priors.lib.joint_snapshot_preflight import validate_joint_snapshot
from r0.modules.priors.lib.raw_draw_records import RawDrawRecord
from r0.modules.priors.lib.topology_hyperprior_decoding import complete_topology_record


def _tuples(value: Any) -> Any:
    if type(value) in (list, tuple):
        return tuple(_tuples(v) for v in cast(list[Any] | tuple[Any, ...], value))
    return value


def _raw_draw(value: Any) -> RawDrawRecord:
    raw = complete_topology_record(value, RawDrawRecord)
    for key in ("owner", "shape", "parameters", "values"):
        raw[key] = _tuples(raw[key])
    return RawDrawRecord(**raw)


def preflight_topology_templates(value: Any, *, limits: CausalLimits, state_bytes: int) -> None:
    if type(value) in (list, tuple):
        for item in cast(list[Any] | tuple[Any, ...], value):
            preflight_topology_templates(item, limits=limits, state_bytes=state_bytes)
        return
    if type(value) is not dict:
        return
    raw = cast(dict[str, Any], value)
    if "payload" in raw and "engine" in raw:
        snapshot = GeneratorSnapshot(**complete_topology_record(raw, GeneratorSnapshot))
        validate_joint_snapshot(snapshot, byte_limit=state_bytes)
        return
    if "state_before" in raw and "distribution" in raw:
        _raw_draw(raw)
        return
    for item in raw.values():
        preflight_topology_templates(item, limits=limits, state_bytes=state_bytes)
    if raw.get("version") == "sampled-feature-conv1d-r0-v1":
        fields = complete_topology_record(raw, SampledFeatureConv1D)
        request = feature_conv1d_request_from_dict(fields["request"], limits=limits)
        function = decode_feature_conv1d(fields["function"])
        initial = tuple(
            GeneratorSnapshot(**complete_topology_record(s, GeneratorSnapshot))
            for s in fields["initial"]
        )
        final = tuple(
            GeneratorSnapshot(**complete_topology_record(s, GeneratorSnapshot))
            for s in fields["final"]
        )
        sample = SampledFeatureConv1D(
            request,
            function,
            tuple(tuple(c) for c in fields["choices"]),
            tuple(_raw_draw(r) for r in fields["raw"]),
            initial,
            final,
            fields["version"],
        )
        if (
            function.layout != request.layout
            or function.input_length != request.input_length
            or function.input_channels != request.input_channels
        ):
            raise ValueError("topology template convolution layout/input configuration differs")
        if len(function.layers) != len(request.layers):
            raise ValueError("topology template convolution layer count differs")
        for choice, prior, layer in zip(
            sample.choices, request.layers, function.layers, strict=True
        ):
            geometry, activation, law = choice
            if (
                geometry >= len(prior.geometries)
                or activation >= len(prior.activations)
                or law >= len(prior.kernel_laws)
            ):
                raise ValueError("topology template convolution choice is outside its catalog")
            if (
                layer.geometry != prior.geometries[geometry]
                or layer.activation != prior.activations[activation]
                or layer.placement != prior.placement
                or layer.activation_scope != prior.activation_scope
            ):
                raise ValueError(
                    "topology template convolution chosen geometry/configuration differs"
                )
