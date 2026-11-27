"""Atomic statistics sampling; row innovations remain separately keyed at generation."""

from dataclasses import asdict

import numpy as np

from r0.modules.priors.lib.function_metadata import admit_function_metadata
from r0.modules.priors.lib.generator_snapshot import snapshot_generator
from r0.modules.priors.lib.network_innovation_compile import compile_network_innovations
from r0.modules.priors.lib.network_innovation_plan import (
    NetworkInnovationPlan,
    NetworkInnovationStage,
)
from r0.modules.priors.lib.network_innovation_recipe import with_network_innovations
from r0.modules.priors.lib.network_innovation_sampling_records import (
    NetworkInnovationRequest,
    SampledNetworkInnovations,
)
from r0.modules.priors.lib.network_recipe import NetworkRecipe
from r0.modules.priors.lib.network_scalar_priors import (
    NetworkScalarValue,
    realize_scalar,
    resolve_scalar_law,
)
from r0.modules.priors.lib.raw_draw_capture import RawDrawLedger, raw_transactions


def sample_innovation_record(
    source: NetworkRecipe, request: NetworkInnovationRequest, rng: np.random.Generator
) -> SampledNetworkInnovations:
    if type(source) is not NetworkRecipe or type(request) is not NetworkInnovationRequest:
        raise ValueError("statistics sampling requires frozen source and request")
    selected = [
        (i, b.output_width)
        for i, b in enumerate(source.function.blocks)
        if not (request.exempt_first and i == 0)
    ]
    if not selected:
        raise ValueError("noise request has no participating blocks; unused laws rejected")
    probe = NetworkInnovationPlan(
        tuple(
            None
            if request.exempt_first and i == 0
            else NetworkInnovationStage(
                (0.0,) * b.output_width, (1.0,) * b.output_width, request.placement
            )
            for i, b in enumerate(source.function.blocks)
        ),
        (1.0,) * source.function.input_width if request.importance is not None else (),
        request.importance_gates,
        request.profile,
    )
    compile_network_innovations(source, probe)
    count = 2 * sum(w for _, w in selected) + source.function.input_width
    raw = request.raw_limits
    # Scalar TNLU uses three draws. Charge full metadata, scalar records, half-normals,
    # and RNG snapshots before any statistics draw, then exact-check the final object.
    extra = 32768 * count + 4 * raw.state_bytes
    if (
        4 * count > raw.draws
        or 4 * count > raw.numbers
        or extra > raw.working_bytes
        or extra > source.limits.working_bytes
        or 512 * count > source.limits.parameters
    ):
        raise ValueError("network innovation statistics admission exceeded before draws")
    initial = snapshot_generator(rng, byte_limit=raw.state_bytes)
    ledger = RawDrawLedger(limits=raw)
    with ledger.transactions({"statistics": rng}) as (owned, capture):
        stream = owned["statistics"]
        records: list[NetworkScalarValue] = []
        scale_law = resolve_scalar_law(request.scale, stream, capture, "noise-scale")
        mean_law = resolve_scalar_law(request.mean, stream, capture, "noise-mean")
        cache: dict[tuple[str, int, int], float] = {}

        def value(kind: str, block: int, channel: int) -> float:
            scope = request.scale_scope if kind == "scale" else request.mean_scope
            key = (
                kind,
                block if scope != "shared" else -1,
                channel if scope == "per-channel" else -1,
            )
            if key not in cache:
                result = realize_scalar(
                    scale_law if kind == "scale" else mean_law,
                    stream,
                    capture,
                    f"{kind}:{key[1]}:{key[2]}",
                )
                records.append(result)
                cache[key] = result.value
            return cache[key]

        stages: list[NetworkInnovationStage | None] = []
        for i, block in enumerate(source.function.blocks):
            if request.exempt_first and i == 0:
                stages.append(None)
                continue
            scales = tuple(value("scale", i, j) for j in range(block.output_width))
            if any(v < 0 for v in scales):
                raise ValueError("sampled noise scale is negative; no retry")
            if request.channel_scale == "half-normal":
                draws = capture.draw(
                    stream,
                    owner=("network-innovations", f"block:{i}"),
                    role="channel-scale",
                    distribution="normal",
                    shape=(block.output_width,),
                )
                scales = tuple(
                    float(abs(z) * scale) for z, scale in zip(draws, scales, strict=True)
                )
            stages.append(
                NetworkInnovationStage(
                    tuple(value("mean", i, j) for j in range(block.output_width)),
                    scales,
                    request.placement,
                )
            )
        importance: tuple[float, ...] = ()
        if request.importance is not None:
            law = resolve_scalar_law(request.importance, stream, capture, "importance")
            vals = [
                realize_scalar(law, stream, capture, f"importance:{i}")
                for i in range(source.function.input_width)
            ]
            records.extend(vals)
            importance = tuple(v.value for v in vals)
        plan = NetworkInnovationPlan(
            tuple(stages), importance, request.importance_gates, request.profile
        )
        # Construct/admit full runtime before publishing the statistics stream.
        with_network_innovations(source, plan)
        result = SampledNetworkInnovations(
            request,
            plan,
            tuple(records),
            capture.freeze(),
            initial,
            snapshot_generator(stream, byte_limit=raw.state_bytes),
        )
        admit_function_metadata(asdict(result), source.limits)
    return result


def sample_network_innovations(
    source: NetworkRecipe, request: NetworkInnovationRequest, *, statistics_rng: np.random.Generator
):
    # Keep the outer transaction open through combined source+runtime+raw admission.
    with raw_transactions({"statistics": statistics_rng}, None) as (streams, _):
        sample = sample_innovation_record(source, request, streams["statistics"])
        return with_network_innovations(source, sample.plan, sampling=sample)
