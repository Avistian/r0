"""Three owned streams bind scalar laws to actual frozen neural architectures."""

from dataclasses import asdict, replace

import numpy as np

from r0.modules.priors.lib.function_metadata import admit_function_metadata
from r0.modules.priors.lib.generator_snapshot import snapshot_generator
from r0.modules.priors.lib.network_hyperprior_records import (
    NetworkHyperpriorRequest,
    SampledHyperpriorNetwork,
)
from r0.modules.priors.lib.network_sampling import sample_network
from r0.modules.priors.lib.network_scalar_priors import (
    NetworkScalarValue,
    realize_scalar,
    resolve_scalar_law,
)
from r0.modules.priors.lib.raw_draw_capture import RawDrawLedger, raw_transactions


def sample_hyperprior_network(
    request: NetworkHyperpriorRequest,
    *,
    hyper_rng: np.random.Generator,
    named_rng: np.random.Generator,
    raw_rng: np.random.Generator,
) -> SampledHyperpriorNetwork:
    if type(request) is not NetworkHyperpriorRequest:
        raise ValueError("profile sampling requires a frozen hyperprior request")
    request.admit()
    with raw_transactions({"hyper": hyper_rng, "named": named_rng, "raw": raw_rng}, None) as (
        streams,
        _,
    ):
        hyper_initial = snapshot_generator(
            streams["hyper"], byte_limit=request.template.raw_limits.state_bytes
        )
        ledger = RawDrawLedger(limits=request.template.raw_limits)
        with ledger.transactions({"hyper": streams["hyper"]}) as (owned, capture):
            rng = owned["hyper"]
            records: list[NetworkScalarValue] = []
            depth_law = resolve_scalar_law(request.depth, rng, capture, "depth")
            depth = realize_scalar(depth_law, rng, capture, "depth")
            records.append(depth)
            hidden = depth.value - (1 if request.depth_semantics == "total-blocks" else 0)
            if not 0 <= hidden <= request.max_hidden_blocks or hidden != round(hidden):
                raise ValueError(
                    "sampled hidden-block count exceeds operational cap or is not integer; no retry"
                )
            widths: list[int] = []
            if hidden:
                width_law = resolve_scalar_law(request.width, rng, capture, "width")
                for index in range(1 if request.width_scope == "shared" else int(hidden)):
                    value = realize_scalar(width_law, rng, capture, f"width:{index}")
                    records.append(value)
                    if not 1 <= value.value <= request.max_hidden_width or value.value != round(
                        value.value
                    ):
                        raise ValueError(
                            "sampled hidden width exceeds operational cap "
                            "or is not integer; no retry"
                        )
                    widths.append(int(value.value))
                if request.width_scope == "shared":
                    widths *= int(hidden)
            weight = realize_scalar(
                resolve_scalar_law(request.weight_scale, rng, capture, "weight"),
                rng,
                capture,
                "weight",
            )
            records.append(weight)
            bias = request.template.bias_scale
            if request.bias_scale is not None:
                chosen = realize_scalar(
                    resolve_scalar_law(request.bias_scale, rng, capture, "bias"),
                    rng,
                    capture,
                    "bias",
                )
                records.append(chosen)
                bias = chosen.value
            effective = replace(
                request.template,
                widths=(*widths, *request.template.widths),
                weight_scale=weight.value,
                bias_scale=bias,
            )
        inner = sample_network(effective, named_rng=streams["named"], raw_rng=streams["raw"])
        result = SampledHyperpriorNetwork(
            inner.request,
            inner.function,
            inner.choices,
            inner.raw,
            inner.initial,
            inner.final,
            profile=request,
            parameters=tuple(records),
            hyper_raw=ledger.freeze(),
            hyper_initial=hyper_initial,
            hyper_final=snapshot_generator(
                streams["hyper"], byte_limit=request.template.raw_limits.state_bytes
            ),
        )
        admit_function_metadata(asdict(result), request.template.limits)
        return result
