"""Atomic named architecture and raw weight draws; no training or row-wise dropout."""

import math
from dataclasses import asdict
from typing import Any

import numpy as np
from numpy.typing import NDArray

from r0.modules.priors.lib.causal_activations import CausalActivation
from r0.modules.priors.lib.generator_snapshot import snapshot_generator
from r0.modules.priors.lib.network_sampling_records import NetworkSamplingRequest, SampledNetwork
from r0.modules.priors.lib.raw_draw_capture import RawDrawLedger, raw_transactions
from r0.modules.priors.lib.structural_network import (
    FrozenNetworkFunction,
    NetworkAffine,
    NetworkBlock,
    NetworkProjection,
)


def sample_network(
    request: NetworkSamplingRequest, *, named_rng: np.random.Generator, raw_rng: np.random.Generator
) -> SampledNetwork:
    if type(request) is not NetworkSamplingRequest:
        raise ValueError("network sampling requires its frozen request")
    request.admit()
    ledger = RawDrawLedger(limits=request.raw_limits)
    with raw_transactions({"named": named_rng, "raw": raw_rng}, ledger) as (streams, capture):
        assert capture is not None
        initial = tuple(
            snapshot_generator(r, byte_limit=request.raw_limits.state_bytes)
            for r in streams.values()
        )
        choices: list[tuple[str, str]] = []

        def draw(
            role: str,
            law: str,
            shape: tuple[int, ...] = (),
            parameters: tuple[tuple[str, Any], ...] = (),
            named: bool = False,
        ) -> NDArray[Any]:
            return capture.draw(
                streams["named" if named else "raw"],
                owner=("network",),
                role=role,
                distribution=law,
                shape=shape,
                parameters=parameters,
            )

        def select[T](role: str, options: tuple[T, ...]) -> T:
            slot = int(draw(role, "integers", parameters=(("high", len(options)),), named=True))
            choices.append((role, str(slot)))
            return options[slot]

        family = select("family", request.families)
        dropout = request.dropout
        if request.dropout_profile == "p1-beta":
            a = float(
                draw("dropout-a", "uniform", parameters=(("low", 0.1), ("high", 5.0)), named=True)
            )
            b = float(
                draw("dropout-b", "uniform", parameters=(("low", 0.1), ("high", 5.0)), named=True)
            )
            ga = float(
                draw("dropout-gamma-a", "gamma", parameters=(("concentration", a),), named=True)
            )
            gb = float(
                draw("dropout-gamma-b", "gamma", parameters=(("concentration", b),), named=True)
            )
            if not math.isfinite(ga + gb) or ga + gb <= 0:
                raise ValueError("invalid Beta gamma ratio; not retried")
            dropout = 0.9 * (ga / (ga + gb))
            if not math.isfinite(dropout):
                raise ValueError("nonfinite Beta construction; not retried")
        choices.append(("dropout", repr(dropout)))
        shared = (
            select("shared-activation", request.activations)
            if request.homogeneous and (request.output_activation or len(request.widths) > 1)
            else None
        )
        widths = [request.input_width]
        blocks: list[NetworkBlock] = []
        for index, width in enumerate(request.widths):
            prefix = f"block:{index}"
            if family == "irregular":
                # Empty realized parent sets retain a bias-only function, not resampling.
                candidates = (
                    draw(prefix + "-parents", "random", (len(widths),), named=True)
                    < request.skip_probability
                )
                parents = tuple(i for i, v in enumerate(candidates) if v)
                # A zero masked input edge represents a legal disconnected block.
                disconnected = not parents
                if disconnected:
                    parents = (0,)
            else:
                parents = (index,)
                disconnected = False
            incoming = sum(widths[i] for i in parents)
            expert_count = request.experts if family == "gated" else 1

            def affine(
                name: str,
                ins: int,
                outs: int,
                activation: CausalActivation,
                placement: str,
                masking: bool = True,
                zero: bool = False,
                layer_index: int = index,
            ) -> NetworkAffine:
                law = select(name + "-law", request.weight_laws)
                args = (
                    (("low", -math.sqrt(3)), ("high", math.sqrt(3)))
                    if law == "uniform"
                    else (("df", 3.0),)
                    if law == "student_t"
                    else ()
                )
                matrix = draw(name + "-weights", law, (outs, ins), args)
                scale = request.weight_scale / (math.sqrt(ins) if request.fan_in else 1.0)
                mask = np.ones((outs, ins), dtype=bool)
                if masking and family == "individual-sparse":
                    p = (
                        0.0
                        if request.mask_profile == "p1-individual" and layer_index == 0
                        else dropout
                    )
                    if request.mask_profile == "p1-individual":
                        p = min(p, 0.99)
                        scale = request.weight_scale / (1 - p**request.scale_exponent)
                    mask = draw(name + "-mask", "random", (outs, ins)) >= p
                if masking and family == "block-sparse":
                    count = (
                        select(
                            name + "-blocks",
                            tuple(range(1, math.ceil(math.sqrt(min(ins, outs))) + 1)),
                        )
                        if request.mask_profile == "p1-block"
                        else min(request.block_limit, ins, outs)
                    )
                    mask[:] = False
                    h, w = outs // count, ins // count
                    for k in range(count):
                        mask[k * h : (k + 1) * h, k * w : (k + 1) * w] = True
                    if request.mask_profile == "p1-block":
                        scale = request.weight_scale / (
                            float(mask.mean()) ** request.scale_exponent
                        )
                if zero:
                    mask[:] = False
                if request.mask_profile.startswith("p1-"):
                    bias = draw(
                        name + "-bias",
                        "uniform",
                        (outs,),
                        (("low", -1 / math.sqrt(ins)), ("high", 1 / math.sqrt(ins))),
                    )
                else:
                    bias = draw(name + "-bias", "normal", (outs,)) * request.bias_scale
                return NetworkAffine(
                    tuple(tuple(map(float, row)) for row in matrix),
                    tuple(map(float, bias)),
                    tuple(tuple(map(bool, row)) for row in mask),
                    scale,
                    activation,
                    placement,
                )

            experts: list[NetworkAffine] = []
            for j in range(expert_count):
                name = f"{prefix}-expert:{j}"
                if index == len(request.widths) - 1 and not request.output_activation:
                    activation = CausalActivation()
                else:
                    activation = (
                        shared
                        if shared is not None
                        else select(name + "-activation", request.activations)
                    )
                experts.append(
                    affine(name, incoming, width, activation, request.placement, zero=disconnected)
                )
            gate = (
                affine(
                    prefix + "-gate",
                    incoming,
                    expert_count,
                    CausalActivation(),
                    "after",
                    masking=False,
                )
                if family == "gated"
                else None
            )
            skips: list[NetworkProjection] = []
            if family in ("skip", "residual"):
                sources = (
                    (index,)
                    if family == "residual"
                    else tuple(
                        i
                        for i in range(index)
                        if float(draw(f"{prefix}-skip:{i}-choice", "random", named=True))
                        < request.skip_probability
                    )
                )
                for source in sources:
                    skips.append(
                        NetworkProjection(
                            source,
                            affine(
                                f"{prefix}-skip:{source}",
                                widths[source],
                                width,
                                CausalActivation(),
                                "after",
                                masking=False,
                            ),
                        )
                    )
            blocks.append(NetworkBlock(parents, tuple(experts), gate, tuple(skips)))
            widths.append(width)
        function = FrozenNetworkFunction(request.input_width, tuple(blocks))
        final = tuple(
            snapshot_generator(r, byte_limit=request.raw_limits.state_bytes)
            for r in streams.values()
        )
        result = SampledNetwork(request, function, tuple(choices), capture.freeze(), initial, final)
        # Charge all retained metadata as well as function tensors before publication.
        from r0.modules.priors.lib.function_metadata import admit_function_metadata

        admit_function_metadata(asdict(result), request.limits)
        return result
