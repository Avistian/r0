"""Frozen target references retain the complete matched/fresh comparison environment."""

from dataclasses import asdict, dataclass, fields
from typing import Any, cast

from r0.modules.priors.lib.causal_limits import CausalLimits
from r0.modules.priors.lib.gp_column_metadata import gp_target_metadata
from r0.modules.priors.lib.gp_column_reference import (
    GPColumnReference,
    GPReferenceState,
    evaluate_frozen_gp_reference,
    validate_gp_reference_controls,
)
from r0.modules.priors.lib.gp_target_environment import (
    GPColumnTargetRecipe,
    admit_gp_target_environment,
    gp_column_target_recipe_from_dict,
    gp_target_environment_metadata,
)
from r0.modules.priors.lib.gp_target_links import (
    GPColumnTargetLink,
    effective_gp_target,
    gp_column_target_link_from_dict,
    gp_link_admission,
    gp_target_base,
    gp_target_link_metadata,
    linked_targets,
)
from r0.modules.priors.lib.joint_function_admission import function_record_cost
from r0.modules.priors.lib.representatives import encode, record_fingerprint

_DEFAULT_LIMITS = CausalLimits()


@dataclass(frozen=True, slots=True)
class GPColumnEnvironmentReference(GPColumnReference):
    environment_fingerprint: str = ""
    environment_json: str = ""
    environment_side: str = ""
    mode: str = ""
    fresh_seed: int | None = None
    version: str = "gp-column-environment-reference-v1"


def gp_environment_reference_from_dict(
    data: Any,
    *,
    feature: int,
    anchor_row: int = 0,
    points: int = 129,
    bounds: tuple[float, float] | None = None,
    limits: CausalLimits = _DEFAULT_LIMITS,
) -> GPColumnEnvironmentReference:
    linked = (
        type(data) is dict
        and cast(dict[str, Any], data).get("version") == "gp-column-target-link-v1"
    )
    if linked:
        raw, _, targets = gp_target_link_metadata(data, limits=limits)
        _, task = gp_target_environment_metadata(raw, limits=limits)
    else:
        raw, task = gp_target_environment_metadata(data, limits=limits)
        targets = ()
    c = raw["comparison"]
    from r0.modules.priors.lib.gp_column_decoding import gp_population_metadata

    request = gp_population_metadata(c["original"], limits=limits)
    validate_gp_reference_controls(request, feature, anchor_row, points, bounds)
    target = (
        next(iter(targets))
        if linked
        else gp_target_metadata(
            c["replacement"] if raw["side"] == "replacement" else c["original"]["target"]
        )
    )
    numbers = 8 * points * (request.columns + len(target.weights) + 8 * request.outputs + 2)
    if linked:
        gp_target_link_metadata(data, limits=limits, query_numbers=numbers)
        recipe = gp_column_target_link_from_dict(data, limits=limits)
    else:
        admit_gp_target_environment(
            request, gp_target_metadata(c["replacement"]), task, limits, extra_numbers=numbers
        )
        recipe = gp_column_target_recipe_from_dict(data, limits=limits)
    base = gp_target_base(recipe)
    comparison = base.comparison
    p = comparison.original
    target, centers, scales = effective_gp_target(recipe)
    reference = evaluate_frozen_gp_reference(
        GPReferenceState(
            p.population_id,
            p.seed,
            p.task_id,
            task,
            target,
            comparison.draw.values,
            centers,
            scales,
        ),
        feature=feature,
        anchor_row=anchor_row,
        points=points,
        bounds=bounds,
        limits=limits,
    )
    return GPColumnEnvironmentReference(
        **{
            f.name: getattr(reference, f.name)
            for f in fields(GPColumnReference)
            if f.name != "version"
        },
        environment_fingerprint=record_fingerprint(asdict(recipe)),
        environment_json=encode(asdict(recipe)),
        environment_side="linked-replacement" if linked else base.side,
        mode=comparison.mode,
        fresh_seed=comparison.fresh_seed,
    )


def evaluate_gp_environment_reference(
    recipe: GPColumnTargetRecipe | GPColumnTargetLink,
    *,
    feature: int,
    anchor_row: int = 0,
    points: int = 129,
    bounds: tuple[float, float] | None = None,
    limits: CausalLimits = _DEFAULT_LIMITS,
) -> GPColumnEnvironmentReference:
    base = gp_target_base(recipe)
    validate_gp_reference_controls(base.request, feature, anchor_row, points, bounds)
    target, _, _ = effective_gp_target(recipe)
    numbers = (
        8 * points * (base.request.columns + len(target.weights) + 8 * base.request.outputs + 2)
    )
    gp_link_admission(base, linked_targets(recipe), limits, query_numbers=numbers)
    function_record_cost(recipe, limits)
    return gp_environment_reference_from_dict(
        asdict(recipe),
        feature=feature,
        anchor_row=anchor_row,
        points=points,
        bounds=bounds,
        limits=limits,
    )
