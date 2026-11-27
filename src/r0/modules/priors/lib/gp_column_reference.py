"""Counterfactual frozen-target slices, distinct from conditioning GP feature columns."""

import math
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import torch

from r0.modules.priors.lib.causal_limits import CausalLimits
from r0.modules.priors.lib.function_metadata import admit_function_metadata
from r0.modules.priors.lib.gp_column_decoding import (
    gp_column_recipe_from_dict,
    gp_fields,
    gp_population_metadata,
)
from r0.modules.priors.lib.gp_column_draw import gp_matrix
from r0.modules.priors.lib.gp_column_metadata import gp_sequence
from r0.modules.priors.lib.gp_column_output import GPColumnRecipe
from r0.modules.priors.lib.gp_column_population import GPColumnPopulation, GPColumnRequest
from r0.modules.priors.lib.gp_column_target import GPColumnTarget, evaluate_gp_column_target
from r0.modules.priors.lib.linear import LinearRecipe
from r0.modules.priors.lib.targets import classification_probabilities
from r0.modules.priors.lib.weight_geometry import Matrix

_DEFAULT_LIMITS = CausalLimits()


@dataclass(frozen=True, slots=True)
class GPColumnReference:
    population_id: str
    seed: int
    task_id: int
    feature: int
    anchor_row: int
    anchor_inputs: tuple[float, ...]
    bounds: tuple[float, float]
    range_source: str
    values: tuple[float, ...]
    target: GPColumnTarget
    centers: tuple[float, ...]
    scales: tuple[float, ...]
    raw_scores: Matrix
    scores: Matrix
    probabilities: Matrix | None
    regression_mean: tuple[float, ...] | None
    observation_noise_std: float | None
    version: str = "gp-column-target-reference-v1"


def _admit(
    request: GPColumnRequest,
    feature: int,
    anchor_row: int,
    points: int,
    bounds: tuple[float, float] | None,
    limits: CausalLimits,
) -> None:
    if (
        type(feature) is not int
        or not 0 <= feature < request.columns
        or type(anchor_row) is not int
        or not 0 <= anchor_row < request.rows
        or type(points) is not int
        or not 2 <= points <= 512
        or bounds is not None
        and (
            type(bounds) is not tuple
            or len(bounds) != 2
            or any(type(v) not in (int, float) or not math.isfinite(v) for v in bounds)
            or bounds[0] > bounds[1]
        )
    ):
        raise ValueError("invalid GP target reference feature, row, points or bounds")
    # Retained source plus strict replay and query grid/raw/normalized/probability
    # buffers coexist. This is a conservative reservation, not measured RSS.
    channels = request.columns + request.hidden + 8 * request.outputs + 2
    numbers = 8 * points * channels
    for ceiling in (request.limits, limits):
        report = request.admit(ceiling)
        if (
            4 * report["parameters"] + numbers > ceiling.parameters
            or 4 * report["estimated_working_bytes"] + 128 * numbers > ceiling.working_bytes
        ):
            raise ValueError("GP target reference aggregate limit exceeded")


def gp_column_reference_from_dict(
    data: Any,
    *,
    feature: int,
    anchor_row: int = 0,
    points: int = 129,
    bounds: tuple[float, float] | None = None,
    limits: CausalLimits = _DEFAULT_LIMITS,
) -> GPColumnReference:
    """Validate controls and aggregate work before replaying a realized source."""
    admit_function_metadata(data, limits)
    raw = gp_fields(data, GPColumnRecipe)
    request = gp_population_metadata(raw["source"], limits=limits)
    task = gp_fields(raw["task"], LinearRecipe)
    gp_sequence(task["class_offsets"])
    task["class_offsets"] = tuple(task["class_offsets"])
    adapter = LinearRecipe(**task)
    GPColumnRecipe(request, adapter, raw["version"])
    _admit(request, feature, anchor_row, points, bounds, limits)
    recipe = gp_column_recipe_from_dict(data, limits=limits)
    assert isinstance(recipe.source, GPColumnPopulation)
    population = recipe.source
    anchor = population.draw.values[anchor_row]
    if bounds is None:
        column = tuple(row[feature] for row in population.draw.values)
        low, high = min(column), max(column)
    else:
        low, high = bounds
    x = np.repeat(np.asarray(anchor, dtype=np.float64)[None, :], points, axis=0)
    fraction = np.linspace(0, 1, points)
    values = (1 - fraction) * low + fraction * high
    x[:, feature] = values
    raw_scores = evaluate_gp_column_target(population.target, gp_matrix(x), limits=limits)
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        normalized = (np.asarray(raw_scores) - population.centers) / population.scales
    if not np.isfinite(normalized).all():
        raise ValueError("nonfinite GP target reference; no clipping or refit")
    probabilities = None
    mean = None
    noise = None
    if adapter.task_type == "regression":
        mean = tuple(float(v) for v in normalized[:, 0])
        noise = adapter.noise_ratio
    else:
        probabilities = gp_matrix(
            classification_probabilities(
                adapter, torch.tensor(normalized, dtype=torch.float64)
            ).numpy()
        )
    return GPColumnReference(
        population.population_id,
        population.seed,
        population.task_id,
        feature,
        anchor_row,
        anchor,
        (float(low), float(high)),
        "observed_min_max" if bounds is None else "explicit",
        tuple(float(v) for v in values),
        population.target,
        population.centers,
        population.scales,
        raw_scores,
        gp_matrix(normalized),
        probabilities,
        mean,
        noise,
    )


def evaluate_gp_column_reference(
    recipe: GPColumnRecipe,
    *,
    feature: int,
    anchor_row: int = 0,
    points: int = 129,
    bounds: tuple[float, float] | None = None,
    limits: CausalLimits = _DEFAULT_LIMITS,
) -> GPColumnReference:
    """Vary one column at a retained row, with original target/calibration fixed.

    This is a target-equation counterfactual, not a conditional feature posterior
    or a fresh joint GP population. No query fitting, label or noise draws occur.
    """
    if type(recipe) is not GPColumnRecipe or type(recipe.source) is not GPColumnPopulation:
        raise ValueError("GP target reference requires a retained realized population")
    _admit(recipe.request, feature, anchor_row, points, bounds, limits)
    return gp_column_reference_from_dict(
        asdict(recipe),
        feature=feature,
        anchor_row=anchor_row,
        points=points,
        bounds=bounds,
        limits=limits,
    )
