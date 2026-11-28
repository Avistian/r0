"""Attach source activation preferences to sampled Gaussian structural equations."""

from dataclasses import asdict, fields

import numpy as np

from r0.modules.priors.lib.causal import CausalRecipe, causal_rng
from r0.modules.priors.lib.causal_nodes import CausalNode
from r0.modules.priors.lib.parameter_preferences import ParameterSampler
from r0.modules.priors.lib.preference_limits import PreferenceLimits
from r0.modules.priors.lib.representatives import record_fingerprint
from r0.modules.priors.lib.source_activation_sampling import SourceActivationSampler
from r0.modules.priors.lib.source_causal import SourceCausalRecipe
from r0.modules.priors.lib.source_causal_calibration import fit_source_nodes
from r0.modules.priors.lib.source_causal_prior import (
    SourceActivationPrior,
    SourceStageBinding,
    rescale_request,
    source_stages,
)

_DEFAULT_PRIOR = SourceActivationPrior()
_DEFAULT_PREFERENCES = PreferenceLimits()


def with_source_activations(
    recipe: CausalRecipe,
    *,
    prior: SourceActivationPrior = _DEFAULT_PRIOR,
    seed: int,
    task_id: int,
    numerical_limits: PreferenceLimits = _DEFAULT_PREFERENCES,
    categorical_limits: PreferenceLimits = _DEFAULT_PREFERENCES,
) -> SourceCausalRecipe:
    """Sample once, fit independent ancestors, freeze and validate every stage."""
    if (
        type(recipe) is not CausalRecipe
        or recipe.construction != "gaussian-affine-v1"
        or any(type(n) is not CausalNode for n in recipe.nodes)
    ):
        raise ValueError("source activation sampling requires a base Gaussian-affine recipe")
    if type(prior) is not SourceActivationPrior:
        raise ValueError("source activation sampling requires its frozen prior")
    if any(type(i) is not int or not 0 <= i < 2**63 for i in (seed, task_id)):
        raise ValueError("source sampling identities must lie in [0,2**63)")
    rng = np.random.default_rng(
        int(record_fingerprint(("source-activation-prior-v1", asdict(prior), seed, task_id)), 16)
    )
    sampler = SourceActivationSampler(
        rng,
        mode=prior.mode,
        policy=prior.policy,
        numerical_limits=numerical_limits,
        categorical_limits=categorical_limits,
    )
    rescales = ParameterSampler(rng, default_mode=prior.mode, limits=numerical_limits)
    bindings: list[SourceStageBinding] = []
    parameter_cost = 0
    largest_activation = 0
    for index, (node_index, stage) in enumerate(
        source_stages(tuple(n.parents for n in recipe.nodes), prior)
    ):
        activation = sampler.sample(prior.selector)
        rescale = None
        if prior.selector == "standard":
            rescales.numerical(rescale_request())
            rescale = index
            width = (
                len(recipe.nodes[node_index].weights)
                if stage == "input"
                else len(recipe.nodes[node_index].bias)
            )
            parameter_cost += 6 * width + 2
            largest_activation = max(largest_activation, width)
        parameter_cost += activation.parameter_count
        bindings.append(SourceStageBinding(node_index, stage, index, rescale))
    selections, scales = sampler.freeze(), rescales.freeze()
    noise_units = tuple(float(u) for u in rng.random(len(recipe.nodes)))
    frozen_bindings = tuple(bindings)
    metadata = (
        4
        + len(noise_units)
        + selections.parameter_count
        + scales.parameter_count
        + sum(b.parameter_count for b in frozen_bindings)
    )
    recipe.limits.admit_dimensions(
        tuple(len(n.bias) for n in recipe.nodes),
        tuple(n.parents for n in recipe.nodes),
        rows=recipe.task.rows,
        calibration_rows=max(
            recipe.calibration_rows if recipe.normalize else 0,
            prior.calibration_rows if prior.selector == "standard" else 0,
        ),
        features=recipe.task.features,
        outputs=len(recipe.target),
        extra_parameters=metadata + parameter_cost,
        extra_working_channels=10 * largest_activation if prior.selector == "standard" else 0,
    )
    nodes = fit_source_nodes(
        recipe.nodes,
        prior,
        selections,
        scales,
        frozen_bindings,
        noise_units,
        lambda *keys: causal_rng(seed, task_id, "source-activation-calibration-v1", *keys),
        f"source-activation-calibration-v1/{seed}/{task_id}",
        recipe.limits,
    )
    values = {f.name: getattr(recipe, f.name) for f in fields(CausalRecipe)}
    values.update(
        nodes=nodes,
        version="causal-v2",
        construction="source-activations-v1",
        source_prior=prior,
        source_selections=selections,
        rescale_preferences=scales,
        source_bindings=frozen_bindings,
        noise_units=noise_units,
        calibration_seed=seed,
        calibration_task_id=task_id,
    )
    return SourceCausalRecipe(**values)
