"""Explicit authored-edit matching policy evaluated inside the guarded worker."""

from typing import Any

from r0.modules import priors


def apply_matching_edit_policy(
    recipe: priors.Recipe, data: dict[str, Any], limits: priors.CausalLimits
) -> priors.Recipe:
    source = data.get("matching_source")
    policy = data.get("matching_policy")
    if source is None:
        if policy is not None:
            raise ValueError("matching edit policy requires an original matched view")
        return recipe
    original = priors.recipe_from_dict(source, limits=limits)
    if not isinstance(original, priors.MatchedNetworkInputView) or original.matching is None:
        raise ValueError("matching edit requires a retained matched source")
    if policy not in ("raw", "recalibrate") or type(recipe) is not priors.NetworkInputView:
        raise ValueError("matching edit requires an explicit policy and raw authored view draft")
    if policy == "raw":
        return recipe
    previous = original.matching
    matching = priors.fit_network_variance_match(
        recipe.base,
        before_inputs=previous.before_inputs,
        after_inputs=recipe.active_inputs,
        rows=previous.rows,
        rng=previous.initial.restore(),
    )
    return priors.MatchedNetworkInputView(
        recipe.base, recipe.task, recipe.active_inputs, matching=matching
    )
