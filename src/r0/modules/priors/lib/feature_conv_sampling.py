"""Atomic named catalogs and owned raw kernels; source laws remain explicit r0 choices."""

import math
from typing import Any

import numpy as np

from r0.modules.priors.lib.feature_conv_evaluation import admit_feature_conv1d
from r0.modules.priors.lib.feature_conv_records import FeatureConv1DLayer, FrozenFeatureConv1D
from r0.modules.priors.lib.feature_conv_sampling_records import (
    FeatureConv1DRequest,
    SampledFeatureConv1D,
)
from r0.modules.priors.lib.generator_snapshot import snapshot_generator
from r0.modules.priors.lib.joint_function_admission import function_record_cost
from r0.modules.priors.lib.raw_draw_capture import RawDrawLedger, raw_transactions


def sample_feature_conv1d(
    request: FeatureConv1DRequest, *, named_rng: np.random.Generator, raw_rng: np.random.Generator
) -> SampledFeatureConv1D:
    if type(request) is not FeatureConv1DRequest:
        raise ValueError("convolution sampling requires its frozen request")
    request.admit()
    ledger = RawDrawLedger(limits=request.raw_limits)
    with raw_transactions({"named": named_rng, "raw": raw_rng}, ledger) as (streams, capture):
        assert capture is not None
        initial = tuple(
            snapshot_generator(r, byte_limit=request.raw_limits.state_bytes)
            for r in streams.values()
        )
        layers: list[FeatureConv1DLayer] = []
        choices: list[tuple[int, int, int]] = []
        incoming = request.input_channels
        for i, prior in enumerate(request.layers):
            owner = ("feature-conv1d", str(i))

            def select(role: str, count: int, owner: tuple[str, str] = owner) -> int:
                return int(
                    capture.draw(
                        streams["named"],
                        owner=owner,
                        role=role,
                        distribution="integers",
                        shape=(),
                        parameters=(("high", count),),
                    )
                )

            geometry_slot = select("geometry", len(prior.geometries))
            activation_slot = select("activation", len(prior.activations))
            law_slot = select("kernel-law", len(prior.kernel_laws))
            geometry = prior.geometries[geometry_slot]
            choices.append((geometry_slot, activation_slot, law_slot))

            def draw(
                role: str,
                law: str,
                shape: tuple[int, ...],
                owner: tuple[str, str] = owner,
                student_df: float = prior.student_df,
            ):
                parameters: tuple[tuple[str, Any], ...] = ()
                if law == "uniform":
                    parameters = (("low", -1.0), ("high", 1.0))
                elif law == "student_t":
                    parameters = (("df", student_df),)
                elif law == "rademacher":
                    parameters = (("high", 2),)
                raw = capture.draw(
                    streams["raw"],
                    owner=owner,
                    role=role,
                    distribution="integers" if law == "rademacher" else law,
                    shape=shape,
                    parameters=parameters,
                )
                return 2.0 * raw - 1.0 if law == "rademacher" else raw

            fan = (incoming // geometry.groups) * geometry.kernel_size
            kernel_raw = draw(
                "kernel",
                prior.kernel_laws[law_slot],
                (geometry.output_channels, incoming // geometry.groups, geometry.kernel_size),
            )
            bias_raw = draw("bias", prior.bias_law, (geometry.output_channels,))
            with np.errstate(over="ignore", invalid="ignore"):
                kernel = prior.kernel_mean + kernel_raw * (
                    prior.kernel_scale / math.sqrt(fan) if prior.fan_in else prior.kernel_scale
                )
                bias = prior.bias_mean + bias_raw * prior.bias_scale
            layers.append(
                FeatureConv1DLayer(
                    geometry,
                    tuple(
                        tuple(tuple(float(v) for v in taps) for taps in output) for output in kernel
                    ),
                    tuple(map(float, bias)),
                    prior.activations[activation_slot],
                    prior.placement,
                    prior.activation_scope,
                )
            )
            incoming = geometry.output_channels
        function = FrozenFeatureConv1D(
            request.input_length, request.input_channels, tuple(layers), request.layout
        )
        admit_feature_conv1d(function, 16, limits=request.limits)
        final = tuple(
            snapshot_generator(r, byte_limit=request.raw_limits.state_bytes)
            for r in streams.values()
        )
        result = SampledFeatureConv1D(
            request, function, tuple(choices), capture.freeze(), initial, final
        )
        function_record_cost(result, request.limits)
    return result
