"""Worst-case catalog geometry, raw custody and backend workspace before RNG cloning."""

from r0.modules.priors.lib.causal_limits import CausalLimits
from r0.modules.priors.lib.feature_conv_sampling_records import FeatureConv1DRequest
from r0.modules.priors.lib.raw_draw_capture import RawDrawLedger
from r0.modules.priors.lib.raw_draw_spec import RawDrawSpec


def admit_conv_request(request: FeatureConv1DRequest, limits: CausalLimits) -> dict[str, int]:
    if type(limits) is not CausalLimits or len(request.layers) + 1 > limits.nodes:
        raise ValueError("convolution layer count exceeds declared limits")
    widths = [request.input_length * request.input_channels]
    low, high, channels = request.input_length, request.input_length, (request.input_channels,)
    specs: list[RawDrawSpec] = []
    workspace, parameters, metadata = 0, 0, 0
    for i, layer in enumerate(request.layers):
        metadata += 32 * (
            len(layer.geometries) + len(layer.activations) + len(layer.kernel_laws)
        ) + sum(a.parameter_count for a in layer.activations)
        if metadata > limits.parameters:
            raise ValueError("convolution catalog metadata exceeds parameter cap")
        if len(layer.geometries) * len(channels) > limits.parameters:
            raise ValueError("convolution catalog cross-product exceeds declared work bound")
        lowest, highest = 2**63, 0
        maximum_kernel, maximum_bias = 0, 0
        for geometry in layer.geometries:
            for incoming in channels:
                lower, _, _ = geometry.shape(low, incoming)
                upper, left, right = geometry.shape(high, incoming)
                lowest, highest = min(lowest, lower), max(highest, upper)
                workspace = max(
                    workspace,
                    incoming * (high + left + right) + incoming * geometry.kernel_size * upper,
                )
                maximum_kernel = max(
                    maximum_kernel,
                    geometry.output_channels * (incoming // geometry.groups) * geometry.kernel_size,
                )
                maximum_bias = max(maximum_bias, geometry.output_channels)
        low, high = lowest, highest
        channels = tuple(sorted({g.output_channels for g in layer.geometries}))
        widths.append(high * max(channels))
        parameters += maximum_kernel + maximum_bias
        owner = ("feature-conv1d", str(i))
        for role, count in (
            ("geometry", len(layer.geometries)),
            ("activation", len(layer.activations)),
            ("kernel-law", len(layer.kernel_laws)),
        ):
            specs.append(RawDrawSpec(owner, role, "integers", (), (("high", count),)))
        # Uniform has the largest initializer metadata payload. Flattened maxima
        # reserve more numbers, but preserve actual kernel rank in the byte bound.
        specs.extend(
            (
                RawDrawSpec(
                    owner,
                    "kernel",
                    "uniform",
                    (maximum_kernel, 1, 1),
                    (("low", -1.0), ("high", 1.0)),
                ),
                RawDrawSpec(
                    owner, "bias", "uniform", (maximum_bias,), (("low", -1.0), ("high", 1.0))
                ),
            )
        )
    raw = tuple(specs)
    RawDrawLedger(limits=request.raw_limits).admit_specs(raw, stream="named")
    retained = sum(s.retained_bytes("named") for s in raw) + 4 * request.raw_limits.state_bytes
    if retained > request.raw_limits.working_bytes:
        raise ValueError("convolution raw snapshots and records exceed working cap")
    result = limits.admit_dimensions(
        tuple(widths),
        tuple(() for _ in widths),
        rows=16,
        calibration_rows=0,
        features=widths[0],
        outputs=widths[-1],
        extra_parameters=4 * parameters + metadata,
        extra_working_channels=8 * workspace,
    )
    if result["estimated_working_bytes"] + retained > limits.working_bytes:
        raise ValueError("convolution aggregate construction workspace exceeds limit")
    return {**result, "estimated_working_bytes": result["estimated_working_bytes"] + retained}
