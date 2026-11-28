"""Bounded target-edit lineage on one immutable matched/fresh GP environment."""

from dataclasses import asdict, dataclass
from typing import Any, cast

import numpy as np
import torch

from r0.modules.priors.lib.causal_limits import CausalLimits
from r0.modules.priors.lib.function_metadata import admit_function_metadata
from r0.modules.priors.lib.gp_column_decoding import gp_fields, gp_payload_equal
from r0.modules.priors.lib.gp_column_draw import gp_matrix
from r0.modules.priors.lib.gp_column_metadata import (
    gp_matrix_metadata,
    gp_target_metadata,
    gp_vector_metadata,
)
from r0.modules.priors.lib.gp_column_population import GPColumnRequest
from r0.modules.priors.lib.gp_column_target import GPColumnTarget, evaluate_gp_column_target
from r0.modules.priors.lib.gp_target_environment import (
    GPColumnTargetRecipe,
    admit_gp_target_environment,
    gp_column_target_recipe_from_dict,
    gp_target_adapter_stream,
    gp_target_environment_metadata,
)
from r0.modules.priors.lib.joint_function_admission import function_record_cost
from r0.modules.priors.lib.task_output import complete_task
from r0.modules.priors.lib.training_task import TrainingTask
from r0.modules.priors.lib.weight_geometry import Matrix

_DEFAULT_LIMITS = CausalLimits()
_MAX_LINKS = 8


@dataclass(frozen=True, slots=True)
class GPColumnTargetLink:
    parent: "GPColumnTargetRecipe | GPColumnTargetLink"
    target: GPColumnTarget
    centers: tuple[float, ...]
    scales: tuple[float, ...]
    raw_scores: Matrix
    scores: Matrix
    version: str = "gp-column-target-link-v1"

    @property
    def task(self):
        return gp_target_base(self).task

    @property
    def request(self):
        return gp_target_base(self).request


def gp_target_base(recipe: GPColumnTargetRecipe | GPColumnTargetLink) -> GPColumnTargetRecipe:
    current = recipe
    depth = 0
    while type(current) is GPColumnTargetLink:
        depth += 1
        if depth > _MAX_LINKS:
            raise ValueError("GP target lineage exceeds explicit eight-link limit")
        current = current.parent
    if type(current) is not GPColumnTargetRecipe:
        raise ValueError("GP target lineage must end in an exact retained environment")
    return current


def gp_link_admission(
    base: GPColumnTargetRecipe,
    targets: tuple[GPColumnTarget, ...],
    limits: CausalLimits,
    *,
    query_numbers: int = 0,
) -> None:
    if len(targets) > _MAX_LINKS:
        raise ValueError("GP target lineage exceeds explicit eight-link limit")
    extra = gp_link_extra(base.request, targets, limits) + query_numbers
    admit_gp_target_environment(
        base.request, base.comparison.replacement, base.task, limits, extra_numbers=extra
    )


def gp_link_extra(
    r: GPColumnRequest, targets: tuple[GPColumnTarget, ...], limits: CausalLimits
) -> int:
    extra = 0
    for target in targets:
        if type(target) is not GPColumnTarget or (target.input_width, target.outputs) != (
            r.columns,
            r.outputs,
        ):
            raise ValueError("GP linked target geometry differs from fixed environment")
        channels = target.input_width + len(target.weights) + target.outputs
        if channels > min(r.limits.total_channels, limits.total_channels):
            raise ValueError("GP linked target channel limit exceeded")
        count, storage = function_record_cost(target, limits)
        # Target parameters, fit/input/hidden/output work, retained derived arrays,
        # and serialization/replay copies. No additional GP population is drawn.
        buffers = 8 * (r.rows + r.calibration_rows) * (r.columns + len(target.weights) + r.outputs)
        extra += 4 * (count + buffers) + (4 * storage + 127) // 128 + 128
    return extra


def gp_target_link_metadata(
    data: Any, *, limits: CausalLimits, query_numbers: int = 0
) -> tuple[dict[str, Any], tuple[dict[str, Any], ...], tuple[GPColumnTarget, ...]]:
    admit_function_metadata(data, limits)
    current = data
    layers: list[dict[str, Any]] = []
    targets: list[GPColumnTarget] = []
    while (
        type(current) is dict
        and cast(dict[str, Any], current).get("version") == "gp-column-target-link-v1"
    ):
        if len(layers) >= _MAX_LINKS:
            raise ValueError("GP target lineage exceeds explicit eight-link limit")
        raw = gp_fields(current, GPColumnTargetLink)
        layers.append(raw)
        targets.append(gp_target_metadata(raw["target"]))
        current = raw["parent"]
    base_raw, task = gp_target_environment_metadata(current, limits=limits)
    from r0.modules.priors.lib.gp_column_decoding import gp_population_metadata

    request = gp_population_metadata(base_raw["comparison"]["original"], limits=limits)
    if not layers:
        raise ValueError("GP linked target requires a retained parent environment")
    # Same formula as typed sampling, without constructing/replaying a base.
    for raw, target in zip(layers, targets, strict=True):
        if (target.input_width, target.outputs) != (request.columns, request.outputs):
            raise ValueError("GP linked target geometry differs from fixed environment")
        for key in ("centers", "scales"):
            gp_vector_metadata(raw[key], request.outputs)
        if any(v <= 0 for v in raw["scales"]):
            raise ValueError("GP linked target scales must be positive")
        for key in ("raw_scores", "scores"):
            gp_matrix_metadata(raw[key], request.rows, request.outputs)
    extra = query_numbers + gp_link_extra(request, tuple(targets), limits)
    admit_gp_target_environment(
        request,
        gp_target_metadata(base_raw["comparison"]["replacement"]),
        task,
        limits,
        extra_numbers=extra,
    )
    return base_raw, tuple(layers), tuple(targets)


def fit_gp_target_link(
    parent: GPColumnTargetRecipe | GPColumnTargetLink,
    target: GPColumnTarget,
    *,
    limits: CausalLimits,
) -> GPColumnTargetLink:
    """Fit only on the original independent calibration inputs after full admission."""
    base = gp_target_base(parent)
    c = base.comparison
    calibration = np.asarray(
        evaluate_gp_column_target(target, c.original.calibration_inputs, limits=limits)
    )
    magnitude = np.max(np.abs(calibration), axis=0)
    divisor = np.where(magnitude == 0, 1, magnitude)
    centers = (calibration / divisor).mean(axis=0) * divisor
    scales = (calibration / divisor).std(axis=0) * divisor
    scales = np.where(scales == 0, 1, scales)
    raw = evaluate_gp_column_target(target, c.draw.values, limits=limits)
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        scores = (np.asarray(raw) - centers) / scales
    if (
        not np.isfinite(scores).all()
        or not np.isfinite(centers).all()
        or not np.isfinite(scales).all()
    ):
        raise ValueError("nonfinite GP linked target; no retry or query fit")
    return GPColumnTargetLink(
        parent,
        target,
        tuple(float(v) for v in centers),
        tuple(float(v) for v in scales),
        raw,
        gp_matrix(scores),
    )


def gp_column_target_link_from_dict(
    data: Any, *, limits: CausalLimits = _DEFAULT_LIMITS
) -> GPColumnTargetLink:
    base_raw, _, targets = gp_target_link_metadata(data, limits=limits)
    parent: GPColumnTargetRecipe | GPColumnTargetLink = gp_column_target_recipe_from_dict(
        base_raw, limits=limits
    )
    for target in reversed(targets):
        parent = fit_gp_target_link(parent, target, limits=limits)
    if not isinstance(parent, GPColumnTargetLink) or not gp_payload_equal(asdict(parent), data):
        raise ValueError("GP linked target receipt differs from complete replay")
    return parent


def linked_targets(recipe: GPColumnTargetRecipe | GPColumnTargetLink) -> tuple[GPColumnTarget, ...]:
    gp_target_base(recipe)
    result: list[GPColumnTarget] = []
    while isinstance(recipe, GPColumnTargetLink):
        result.append(recipe.target)
        recipe = recipe.parent
    return tuple(result)


def generate_gp_link(
    recipe: GPColumnTargetLink, *, seed: int, task_id: int, device: str = "cpu"
) -> TrainingTask:
    base = gp_target_base(recipe)
    p = base.comparison.original
    if any(type(v) is not int for v in (seed, task_id)) or (seed, task_id) != (p.seed, p.task_id):
        raise ValueError("GP linked environment requires its exact retained identity")
    if device not in ("cpu", "cuda") or (device == "cuda" and not torch.cuda.is_available()):
        raise ValueError("select available CPU/CUDA device; no fallback")
    gp_link_admission(base, linked_targets(recipe), base.request.limits)
    function_record_cost(recipe, base.request.limits)
    valid = gp_column_target_link_from_dict(asdict(recipe), limits=base.request.limits)
    return complete_task(
        base.task,
        asdict(valid),
        {
            "family": "gp_columns",
            "version": valid.version,
            "kind": "gp-target-edit",
            "lineage": asdict(valid),
            "query_fit": False,
            "lifecycle": "same fixed comparison population and adapter streams",
        },
        torch.tensor(base.comparison.draw.values, dtype=torch.float64, device=device),
        torch.tensor(valid.scores, dtype=torch.float64, device=device),
        seed=seed,
        task_id=task_id,
        rng=lambda *role: gp_target_adapter_stream(base, *role),
    )


def effective_gp_target(recipe: GPColumnTargetRecipe | GPColumnTargetLink):
    if isinstance(recipe, GPColumnTargetLink):
        return recipe.target, recipe.centers, recipe.scales
    c = recipe.comparison
    return (
        (c.replacement, c.centers, c.scales)
        if recipe.side == "replacement"
        else (c.original.target, c.original.centers, c.original.scales)
    )
