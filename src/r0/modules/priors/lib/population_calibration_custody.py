"""Validate retained raw draws and snapshot schemas across both calibration populations."""

from typing import Any, cast

from r0.modules.priors.lib.causal_limits import CausalLimits
from r0.modules.priors.lib.exact_frozen_record import exact_frozen_equal
from r0.modules.priors.lib.feature_conv_decoding import (
    decode_feature_conv1d,
    feature_conv1d_request_from_dict,
)
from r0.modules.priors.lib.feature_conv_sampling_records import SampledFeatureConv1D
from r0.modules.priors.lib.fitted_fit_decoding import decode_fitted_raw_draw
from r0.modules.priors.lib.generator_snapshot import GeneratorSnapshot
from r0.modules.priors.lib.joint_snapshot_preflight import validate_joint_snapshot
from r0.modules.priors.lib.prototype_root_construction import SampledPrototypeRoot
from r0.modules.priors.lib.raw_draw_records import RawDrawLimits
from r0.modules.priors.lib.raw_draw_spec import RawDrawSpec
from r0.modules.priors.lib.recursive_population_decoding import complete_population_record
from r0.modules.priors.lib.sampled_source_root_base import SampledSourceRootBase
from r0.modules.priors.lib.sampled_source_root_profile import SampledSourceRootProfile


def preflight_calibration_custody(data: Any, limits: CausalLimits) -> None:
    def visit(value: Any, bound: int) -> None:
        if type(value) in (tuple, list):
            for child in cast(tuple[Any, ...] | list[Any], value):
                visit(child, bound)
            return
        if type(value) is not dict:
            return
        raw = cast(dict[str, Any], value)
        version = raw.get("version")
        if "raw_limits" in raw:
            caps = RawDrawLimits(**complete_population_record(raw["raw_limits"], RawDrawLimits))
            bound = min(bound, caps.state_bytes)
        if version == "numpy-generator-json-v1" or {"payload", "numpy_version"} & raw.keys():
            snapshot = GeneratorSnapshot(**complete_population_record(raw, GeneratorSnapshot))
            validate_joint_snapshot(snapshot, byte_limit=bound)
        elif version == "raw-draw-v1" or {"state_before", "state_after"} & raw.keys():
            draw = decode_fitted_raw_draw(raw)
            expected = int if draw.dtype == "<i8" else float
            if any(type(v) is not expected for v in draw.values):
                raise ValueError("calibration raw scalar types differ from declared dtype")
        elif version == "sampled-feature-conv1d-r0-v1":
            complete_population_record(raw, SampledFeatureConv1D)
            request = feature_conv1d_request_from_dict(raw["request"], limits=limits)
            function = decode_feature_conv1d(raw["function"])
            choices = tuple(tuple(c) for c in raw["choices"])
            samples = tuple(decode_fitted_raw_draw(r) for r in raw["raw"])
            states = [
                tuple(
                    GeneratorSnapshot(**complete_population_record(s, GeneratorSnapshot))
                    for s in raw[k]
                )
                for k in ("initial", "final")
            ]
            SampledFeatureConv1D(request, function, choices, samples, states[0], states[1])
            if (
                function.input_length != request.input_length
                or function.input_channels != request.input_channels
                or function.layout != request.layout
                or len(function.layers) != len(choices)
            ):
                raise ValueError("calibration convolution geometry differs from request")
            for choice, layer, prior in zip(choices, function.layers, request.layers, strict=True):
                geometry, activation, law = choice
                if (
                    geometry >= len(prior.geometries)
                    or activation >= len(prior.activations)
                    or law >= len(prior.kernel_laws)
                    or not exact_frozen_equal(layer.geometry, prior.geometries[geometry])
                    or not exact_frozen_equal(layer.activation, prior.activations[activation])
                    or layer.placement != prior.placement
                    or layer.activation_scope != prior.activation_scope
                ):
                    raise ValueError("calibration convolution choices differ from frozen layer")
            if any(states[0][i].engine != states[1][i].engine for i in (0, 1)):
                raise ValueError("calibration convolution engines differ")
            previous = {
                key: state.digest for key, state in zip(("named", "raw"), states[0], strict=True)
            }
            engines = {
                key: state.engine for key, state in zip(("named", "raw"), states[0], strict=True)
            }
            numbers = storage = 0
            for i, draw in enumerate(samples):
                if (
                    draw.sequence != i
                    or draw.stream not in previous
                    or draw.generator != engines.get(draw.stream)
                    or draw.state_before != previous.get(draw.stream)
                ):
                    raise ValueError("calibration convolution raw chain differs")
                previous[draw.stream] = draw.state_after
                spec = RawDrawSpec(
                    draw.owner, draw.role, draw.distribution, draw.shape, draw.parameters
                )
                numbers += spec.numbers
                storage += spec.retained_bytes(draw.stream)
            caps = request.raw_limits
            if (
                any(
                    previous[key] != state.digest
                    for key, state in zip(("named", "raw"), states[1], strict=True)
                )
                or len(samples) > caps.draws
                or numbers > caps.numbers
                or storage + 2 * caps.state_bytes > caps.working_bytes
            ):
                raise ValueError(
                    "calibration convolution retained raw budget or final state differs"
                )
        elif version == "sampled-source-root-base-v1":
            complete_population_record(raw, SampledSourceRootBase)
        elif version == "sampled-source-root-profile-v1":
            complete_population_record(raw, SampledSourceRootProfile)
        elif version == "sampled-prototype-root-v1":
            complete_population_record(raw, SampledPrototypeRoot)
        for child in raw.values():
            visit(child, bound)

    visit(data, RawDrawLimits().state_bytes)
