"""Admit the exact prospective GP-query wrapper before replay or new random draws."""

from dataclasses import asdict, replace
from typing import Any

from r0.modules.priors.lib.causal_limits import CausalLimits
from r0.modules.priors.lib.gp_population_metadata_cost import (
    future_gp_root_cost,
    record_cost,
    retained_cost,
)
from r0.modules.priors.lib.gp_root_calibration import GPRootCalibration
from r0.modules.priors.lib.gp_root_query import GPRootCalibrationQuery
from r0.modules.priors.lib.gp_root_query_metadata import admit_query_envelope, query_source_metadata
from r0.modules.priors.lib.weight_geometry import Matrix


def query_construction_plan(
    source: Any, *, seed: int, task_id: int, locations: Matrix | None, limits: CausalLimits
):
    if type(limits) is not CausalLimits:
        raise ValueError("GP query requires explicit caller limits")
    if any(type(v) is not int or not 0 <= v < 2**63 for v in (seed, task_id)):
        raise ValueError("GP query identity requires exact nonnegative integers")
    metadata = query_source_metadata(source, limits=limits)
    if (seed, task_id) in metadata.identities:
        raise ValueError("GP query cannot reuse an original, fit or ancestral identity")
    cap = metadata.limits
    request = replace(
        metadata.request, locations=metadata.request.locations if locations is None else locations
    )
    request.admit(cap)
    future = record_cost(
        GPRootCalibrationQuery,
        dict(
            source=retained_cost(source, cap),
            original=future_gp_root_cost(
                request, metadata.current["base"], metadata.current["bindings"], cap
            ),
            limits=retained_cost(cap, cap),
            version=retained_cost("gp-root-calibration-query-v1", cap),
        ),
    )
    admit_query_envelope(future, (*metadata.populations, (request, metadata.base)), cap)
    return metadata, request


def preflight_gp_root_calibration_query(
    source: Any,
    *,
    seed: int,
    task_id: int,
    locations: Matrix | None = None,
    limits: CausalLimits | None = None,
) -> CausalLimits:
    typed = type(source) in (GPRootCalibration, GPRootCalibrationQuery)
    cap = (
        source.limits if typed and limits is None else CausalLimits() if limits is None else limits
    )
    if type(cap) is not CausalLimits:
        raise ValueError("GP query requires explicit caller limits")
    retained_cost(source, cap)
    raw = asdict(source) if typed else source
    metadata, _ = query_construction_plan(
        raw, seed=seed, task_id=task_id, locations=locations, limits=cap
    )
    return metadata.limits
