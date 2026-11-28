"""Complete-field convolution archives, bounded before reconstruction and exact replay."""

import json
from dataclasses import asdict
from typing import Any

from r0.modules.priors.lib.causal_limits import CausalLimits
from r0.modules.priors.lib.feature_conv_evaluation import admit_feature_conv1d
from r0.modules.priors.lib.feature_conv_records import (
    Conv1DGeometry,
    FeatureConv1DLayer,
    FrozenFeatureConv1D,
)
from r0.modules.priors.lib.feature_conv_sampling import sample_feature_conv1d
from r0.modules.priors.lib.feature_conv_sampling_records import (
    Conv1DLayerPrior,
    FeatureConv1DRequest,
    SampledFeatureConv1D,
)
from r0.modules.priors.lib.function_metadata import admit_function_metadata
from r0.modules.priors.lib.generator_snapshot import GeneratorSnapshot
from r0.modules.priors.lib.network_decoding import activation, array, record
from r0.modules.priors.lib.raw_draw_records import RawDrawLimits

_DEFAULT_LIMITS = CausalLimits()


def _geometry(raw: Any) -> Conv1DGeometry:
    fields = record(raw, Conv1DGeometry)
    fields["explicit_padding"] = array(fields["explicit_padding"])
    return Conv1DGeometry(**fields)


def feature_conv1d_request_from_dict(
    data: Any, *, limits: CausalLimits = _DEFAULT_LIMITS
) -> FeatureConv1DRequest:
    try:
        admit_function_metadata(data, limits)
        fields = record(data, FeatureConv1DRequest)
        priors: list[Conv1DLayerPrior] = []
        for value in array(fields["layers"]):
            prior = record(value, Conv1DLayerPrior)
            prior["geometries"] = tuple(_geometry(g) for g in array(prior["geometries"]))
            prior["activations"] = tuple(activation(a) for a in array(prior["activations"]))
            prior["kernel_laws"] = array(prior["kernel_laws"])
            priors.append(Conv1DLayerPrior(**prior))
        fields["layers"] = tuple(priors)
        fields["limits"] = CausalLimits(**record(fields["limits"], CausalLimits))
        fields["raw_limits"] = RawDrawLimits(**record(fields["raw_limits"], RawDrawLimits))
        result = FeatureConv1DRequest(**fields)
        result.admit(limits=limits)
        return result
    except (KeyError, TypeError, AttributeError, OverflowError, RecursionError) as error:
        raise ValueError("malformed feature convolution request") from error


def feature_conv1d_from_dict(
    data: Any, *, limits: CausalLimits = _DEFAULT_LIMITS
) -> FrozenFeatureConv1D | SampledFeatureConv1D:
    try:
        admit_function_metadata(data, limits)
        if data.get("version") == "sampled-feature-conv1d-r0-v1":
            fields = record(data, SampledFeatureConv1D)
            request = feature_conv1d_request_from_dict(fields["request"], limits=limits)
            initial = array(fields["initial"])
            if len(initial) != 2:
                raise ValueError("sampled convolution requires two owned streams")
            streams = [
                GeneratorSnapshot(**record(s, GeneratorSnapshot)).restore(
                    byte_limit=request.raw_limits.state_bytes
                )
                for s in initial
            ]
            result = sample_feature_conv1d(request, named_rng=streams[0], raw_rng=streams[1])
            if json.dumps(asdict(result), sort_keys=True, allow_nan=False) != json.dumps(
                data, sort_keys=True, allow_nan=False
            ):
                raise ValueError("convolution archive differs from exact complete raw replay")
            return result
        function = decode_feature_conv1d(data)
        admit_feature_conv1d(function, 16, limits=limits)
        return function
    except (KeyError, TypeError, AttributeError, OverflowError, RecursionError) as error:
        raise ValueError("malformed feature convolution archive") from error


def decode_feature_conv1d(data: Any) -> FrozenFeatureConv1D:
    """Decode authored kernels after caller metadata admission, without allocating tensors."""
    try:
        fields = record(data, FrozenFeatureConv1D)
        layers: list[FeatureConv1DLayer] = []
        for value in array(fields["layers"]):
            layer = record(value, FeatureConv1DLayer)
            layer["geometry"] = _geometry(layer["geometry"])
            layer["kernel"] = tuple(
                tuple(array(taps) for taps in array(output)) for output in array(layer["kernel"])
            )
            layer["bias"] = array(layer["bias"])
            layer["activation"] = activation(layer["activation"])
            layers.append(FeatureConv1DLayer(**layer))
        fields["layers"] = tuple(layers)
        return FrozenFeatureConv1D(**fields)
    except (KeyError, TypeError, AttributeError, OverflowError, RecursionError) as error:
        raise ValueError("malformed feature convolution archive") from error
