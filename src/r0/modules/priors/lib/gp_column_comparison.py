"""Target-mechanism edits: matched finite rows or fresh innovations at fixed locations."""

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

from r0.modules.priors.lib.causal_limits import CausalLimits
from r0.modules.priors.lib.function_metadata import admit_function_metadata
from r0.modules.priors.lib.gp_column_decoding import (
    gp_column_population_from_dict,
    gp_fields,
    gp_payload_equal,
    validate_gp_population,
)
from r0.modules.priors.lib.gp_column_draw import GPColumnDraw, build_gp_columns, gp_matrix
from r0.modules.priors.lib.gp_column_population import (
    GPColumnPopulation,
    GPColumnRequest,
    gp_stream,
)
from r0.modules.priors.lib.gp_column_target import GPColumnTarget, evaluate_gp_column_target
from r0.modules.priors.lib.joint_function_admission import function_record_cost
from r0.modules.priors.lib.weight_geometry import Matrix

_DEFAULT_LIMITS = CausalLimits()


@dataclass(frozen=True, slots=True)
class GPColumnComparison:
    original: GPColumnPopulation
    replacement: GPColumnTarget
    mode: str
    fresh_seed: int | None
    draw: GPColumnDraw
    centers: tuple[float, ...]
    scales: tuple[float, ...]
    original_scores: Matrix
    replacement_raw: Matrix
    replacement_scores: Matrix
    version: str = "gp-column-target-edit-v1"


def _admit(request: GPColumnRequest, limits: CausalLimits) -> None:
    for ceiling in (request.limits, limits):
        report = request.admit(ceiling)
        if (
            report["parameters"] * 4 > ceiling.parameters
            or report["estimated_working_bytes"] * 4 > ceiling.working_bytes
        ):
            raise ValueError("GP comparison aggregate population/replay limit exceeded")


def _admit_replacement(
    request: GPColumnRequest, replacement: GPColumnTarget, limits: CausalLimits
) -> None:
    for ceiling in (request.limits, limits):
        count, storage = function_record_cost(replacement, ceiling)
        report = request.admit(ceiling)
        count += (
            8
            * (request.rows + request.calibration_rows)
            * (replacement.input_width + len(replacement.weights) + replacement.outputs)
        )
        storage += 128 * count
        if (
            4 * (count + report["parameters"]) > ceiling.parameters
            or 4 * (storage + report["estimated_working_bytes"]) > ceiling.working_bytes
        ):
            raise ValueError("GP comparison replacement aggregate limit exceeded")


def _validate_controls(
    request: GPColumnRequest, replacement: GPColumnTarget, mode: str, fresh_seed: int | None
) -> None:
    if (replacement.input_width, replacement.outputs) != (
        request.columns,
        request.outputs,
    ):
        raise ValueError("GP comparison replacement geometry differs")
    if (
        mode not in ("matched", "fresh")
        or (mode == "matched" and fresh_seed is not None)
        or (mode == "fresh" and (type(fresh_seed) is not int or not 0 <= fresh_seed < 2**63))
    ):
        raise ValueError("GP comparison requires matched draws or an explicit fresh seed")


def compare_gp_columns(
    original: GPColumnPopulation,
    replacement: GPColumnTarget,
    *,
    mode: str = "matched",
    fresh_seed: int | None = None,
    limits: CausalLimits = _DEFAULT_LIMITS,
) -> GPColumnComparison:
    if (
        type(original) is not GPColumnPopulation
        or type(original.request) is not GPColumnRequest
        or type(replacement) is not GPColumnTarget
    ):
        raise ValueError("GP comparison requires exact population and replacement target")
    _admit(original.request, limits)
    _validate_controls(original.request, replacement, mode, fresh_seed)
    _admit_replacement(original.request, replacement, limits)
    validate_gp_population(original, limits=limits)
    draw = original.draw
    if mode == "fresh":
        rng = gp_stream(original.seed, original.task_id, "fresh-column-comparison", fresh_seed)
        innovations = gp_matrix(rng.normal(size=(original.request.rows, original.request.columns)))
        draw = build_gp_columns(
            original.draw.locations, innovations, kernels=original.draw.kernels, limits=limits
        )
    cal = np.asarray(
        evaluate_gp_column_target(replacement, original.calibration_inputs, limits=limits)
    )
    magnitude = np.max(np.abs(cal), axis=0)
    divisor = np.where(magnitude == 0, 1, magnitude)
    centers = (cal / divisor).mean(axis=0) * divisor
    scales = (cal / divisor).std(axis=0) * divisor
    scales = np.where(scales == 0, 1, scales)
    raw = evaluate_gp_column_target(replacement, draw.values, limits=limits)
    scores = (np.asarray(raw) - centers) / scales
    before = (
        np.asarray(evaluate_gp_column_target(original.target, draw.values, limits=limits))
        - original.centers
    ) / original.scales
    if not np.isfinite(scores).all() or not np.isfinite(before).all():
        raise ValueError("nonfinite GP comparison; no retry")
    return GPColumnComparison(
        original,
        replacement,
        mode,
        fresh_seed,
        draw,
        tuple(float(v) for v in centers),
        tuple(float(v) for v in scales),
        gp_matrix(before),
        raw,
        gp_matrix(scores),
    )


def gp_column_comparison_from_dict(
    data: Any, *, limits: CausalLimits = _DEFAULT_LIMITS
) -> GPColumnComparison:
    from r0.modules.priors.lib.gp_column_decoding import gp_population_metadata
    from r0.modules.priors.lib.gp_column_metadata import (
        gp_draw_metadata,
        gp_matrix_metadata,
        gp_target_metadata,
        gp_vector_metadata,
    )

    admit_function_metadata(data, limits)
    raw = gp_fields(data, GPColumnComparison)
    request = gp_population_metadata(raw["original"], limits=limits)
    _admit(request, limits)
    replacement = gp_target_metadata(raw["replacement"])
    _validate_controls(request, replacement, raw["mode"], raw["fresh_seed"])
    gp_draw_metadata(raw["draw"], request)
    for key in ("centers", "scales"):
        gp_vector_metadata(raw[key], request.outputs)
    if any(v <= 0 for v in raw["scales"]):
        raise ValueError("GP comparison scales differ from replay")
    for key in ("original_scores", "replacement_raw", "replacement_scores"):
        gp_matrix_metadata(raw[key], request.rows, request.outputs)
    _admit_replacement(request, replacement, limits)
    expected = compare_gp_columns(
        gp_column_population_from_dict(raw["original"], limits=limits),
        replacement,
        mode=raw["mode"],
        fresh_seed=raw["fresh_seed"],
        limits=limits,
    )
    if not gp_payload_equal(asdict(expected), data):
        raise ValueError("GP comparison differs from matched/fresh replay")
    return expected
