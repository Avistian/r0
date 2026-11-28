"""Full future calibration envelope and both runtimes before any population replay."""

from dataclasses import fields, replace
from typing import Any

from r0.modules.priors.lib.causal_limits import CausalLimits
from r0.modules.priors.lib.functional_causal import FunctionalCausalRecipe
from r0.modules.priors.lib.functional_target_calibration import FunctionalTargetCalibration
from r0.modules.priors.lib.gp_column_population import GPColumnRequest
from r0.modules.priors.lib.gp_population_metadata_cost import (
    MetadataCost,
    future_gp_root_cost,
    record_cost,
    retained_cost,
    sequence_cost,
)
from r0.modules.priors.lib.gp_root_dags import GPRootDAGRecipe, GPRootRuntime


def admit_gp_calibration_construction(
    original: GPRootDAGRecipe,
    request: GPColumnRequest,
    base: GPRootRuntime,
    limits: CausalLimits,
) -> None:
    original.admit()
    admit_gp_calibration_metadata(
        original,
        original_limits=original.limits,
        original_base=original.base,
        original_request=original.source.request,
        bindings=original.bindings,
        request=request,
        base=base,
        limits=limits,
        fit_metadata=base,
    )


def admit_gp_calibration_metadata(
    original: Any,
    *,
    original_limits: CausalLimits,
    original_base: GPRootRuntime,
    original_request: GPColumnRequest,
    bindings: Any,
    request: GPColumnRequest,
    base: GPRootRuntime,
    limits: CausalLimits,
    fit_metadata: Any,
) -> None:
    """Dominate the final full-envelope and original/fit-pair admission expressions.

    Future GP receipt shapes are exact. The only data-dependent record length is
    constant_channels, reserved at its maximum valid size (the target width).
    Existing eight-copy metadata/four-runtime workspace factors are retained;
    no arbitrary padding, reduced caller budget or population draw is used.
    """
    from r0.modules.priors.lib.gp_root_calibration import GPRootCalibration

    # Strict GP envelope replay preserves declared caps; ordinary graph decoding
    # otherwise tightens them and changes the retained base record.
    for declared, ceiling in (
        (original_base.limits, original_limits),
        (original_limits, limits),
        (base.limits, limits),
    ):
        if any(
            getattr(declared, field.name) > getattr(ceiling, field.name)
            for field in fields(CausalLimits)
        ):
            raise ValueError("GP calibration declared graph/envelope limits exceed replay caps")
    number = MetadataCost(1, 64, 1, 0)
    width = len(original_base.target) if isinstance(original_base, FunctionalCausalRecipe) else 1
    vector = sequence_cost(number, width)
    calibration = record_cost(
        FunctionalTargetCalibration,
        dict(
            rows=number,
            center=vector,
            scale=vector,
            constant_channels=vector,
            sample_digest=retained_cost("0" * 64, limits),
            prefix_fingerprint=retained_cost("0" * 64, limits),
            stream=retained_cost("target-calibration-v1", limits),
            version=retained_cost("functional-target-calibration-v1", limits),
        ),
    )
    envelope = record_cost(
        GPRootCalibration,
        dict(
            original=retained_cost(original, limits),
            fit=future_gp_root_cost(request, fit_metadata, bindings, limits),
            calibration=calibration,
            limits=retained_cost(limits, limits),
            version=retained_cost("gp-root-calibration-v1", limits),
        ),
    )
    for cap in (
        limits,
        original_limits,
        original_request.limits,
        original_base.limits,
        request.limits,
        base.limits,
    ):
        envelope.admit(cap)
        reports = [r.admit(cap) for r in (original_request, request)]
        reports += [replace(graph, limits=cap).admit() for graph in (original_base, base)]
        if (
            envelope.numbers + sum(r["parameters"] for r in reports) > cap.parameters
            or 8 * envelope.storage + 4 * sum(r["estimated_working_bytes"] for r in reports)
            > cap.working_bytes
        ):
            raise ValueError(
                "GP calibration complete future envelope/runtime aggregate exceeds limits"
            )
