"""Admit and freeze source matrix/activation choices before ancestral calibration."""

from dataclasses import asdict, fields

import numpy as np

from r0.modules.priors.lib.causal import CausalRecipe, causal_rng
from r0.modules.priors.lib.causal_activations import CausalActivation
from r0.modules.priors.lib.causal_matrix_sampling import sample_matrix_plan
from r0.modules.priors.lib.causal_nodes import CausalNode, MatrixCausalNode, StochasticCausalNode
from r0.modules.priors.lib.matrix_preference_prior import CausalMatrixPrior
from r0.modules.priors.lib.parameter_preferences import ParameterSampler
from r0.modules.priors.lib.preference_limits import PreferenceLimits
from r0.modules.priors.lib.preferred_contract import Path
from r0.modules.priors.lib.preferred_matrix_sampling import MatrixPlanner
from r0.modules.priors.lib.representatives import record_fingerprint
from r0.modules.priors.lib.source_activation_sampling import SourceActivationSampler
from r0.modules.priors.lib.source_causal_calibration import fit_source_nodes
from r0.modules.priors.lib.source_causal_prior import (
    SOURCE_ACTIVATION_PROFILES,
    SourceActivationPrior,
    SourceStageBinding,
    rescale_request,
    source_stages,
)
from r0.modules.priors.lib.source_matrix import SourceMatrixCausalRecipe

_DEFAULT_PRIOR = SourceActivationPrior()
_DEFAULT_MATRIX_PRIOR = CausalMatrixPrior()
_DEFAULT_LIMITS = PreferenceLimits()


def with_source_matrices(
    recipe: CausalRecipe,
    *,
    prior: SourceActivationPrior = _DEFAULT_PRIOR,
    matrix_prior: CausalMatrixPrior = _DEFAULT_MATRIX_PRIOR,
    seed: int,
    task_id: int,
    numerical_limits: PreferenceLimits = _DEFAULT_LIMITS,
    categorical_limits: PreferenceLimits = _DEFAULT_LIMITS,
) -> SourceMatrixCausalRecipe:
    """One activation scope for matrix and node stages; no task rows enter fitting."""
    if (
        type(recipe) is not CausalRecipe
        or recipe.construction != "gaussian-affine-v1"
        or any(type(n) is not CausalNode for n in recipe.nodes)
    ):
        raise ValueError("joint source sampling requires a base Gaussian-affine recipe")
    if type(prior) is not SourceActivationPrior or type(matrix_prior) is not CausalMatrixPrior:
        raise ValueError("joint source sampling requires frozen source and matrix priors")
    if prior.mode != matrix_prior.mode or prior.policy != matrix_prior.categorical_policy:
        raise ValueError("joint source and matrix preference policies must agree")
    if any(type(i) is not int or not 0 <= i < 2**63 for i in (seed, task_id)):
        raise ValueError("joint source sampling identities must lie in [0,2**63)")
    rng = np.random.default_rng(
        int(
            record_fingerprint(
                ("source-matrix-prior-v1", asdict(prior), asdict(matrix_prior), seed, task_id)
            ),
            16,
        )
    )
    sampler = SourceActivationSampler(
        rng,
        mode=prior.mode,
        policy=prior.policy,
        numerical_limits=numerical_limits,
        categorical_limits=categorical_limits,
    )
    matrix_nodes: list[int] = []

    def activation(path: Path) -> CausalActivation:
        index = path[1]
        assert type(index) is int
        selected = sampler.sample("plain")
        matrix_nodes.append(index)
        return selected

    planner = MatrixPlanner(
        matrix_prior,
        rng,
        activation,
        numerical_limits=numerical_limits,
        categorical_limits=categorical_limits,
    )
    plans = tuple(
        planner.plan(len(n.weights), len(n.bias), ("nodes", i, "weight_construction", "source"))
        if n.parents
        else None
        for i, n in enumerate(recipe.nodes)
    )
    rescales = ParameterSampler(rng, default_mode=prior.mode, limits=numerical_limits)
    bindings: list[SourceStageBinding] = []
    activation_cost = 0
    largest_activation = 0
    for index, (node_index, stage) in enumerate(
        source_stages(tuple(n.parents for n in recipe.nodes), prior)
    ):
        selected = sampler.sample(prior.selector)
        rescale = None
        if prior.selector == "standard":
            rescales.numerical(rescale_request())
            rescale = index
            width = (
                len(recipe.nodes[node_index].weights)
                if stage == "input"
                else len(recipe.nodes[node_index].bias)
            )
            activation_cost += 6 * width + 2
            largest_activation = max(largest_activation, width)
        activation_cost += selected.parameter_count
        bindings.append(SourceStageBinding(node_index, stage, len(matrix_nodes) + index, rescale))
    selections, scales = sampler.freeze(), rescales.freeze()
    numerical, categorical = planner.numerical.freeze(), planner.categorical.freeze()
    matrix_bindings = tuple(planner.bindings)
    noise_units = tuple(float(u) for u in rng.random(len(recipe.nodes)))
    frozen_bindings = tuple(bindings)
    metadata = (
        4
        + len(noise_units)
        + selections.parameter_count
        + scales.parameter_count
        + sum(b.parameter_count for b in frozen_bindings)
        + numerical.parameter_count
        + categorical.parameter_count
        + sum(b.parameter_count for b in matrix_bindings)
        + len(matrix_nodes)
    )
    defaults = StochasticCausalNode().structural_parameter_count
    recipe.limits.admit_dimensions(
        tuple(len(n.bias) for n in recipe.nodes),
        tuple(n.parents for n in recipe.nodes),
        rows=recipe.task.rows,
        calibration_rows=max(
            recipe.admission_calibration_rows,
            prior.calibration_rows if prior.selector == "standard" else 0,
        ),
        features=recipe.task.features,
        outputs=len(recipe.target),
        extra_parameters=metadata
        + activation_cost
        + sum(defaults + p.parameter_count for p in plans if p is not None),
        extra_working_channels=max(
            10 * largest_activation,
            max(
                (
                    StochasticCausalNode.working_channels_for_width(len(n.bias))
                    for n in recipe.nodes
                    if n.parents
                ),
                default=0,
            ),
        ),
    )
    nodes: list[CausalNode] = []
    for node, plan in zip(recipe.nodes, plans, strict=True):
        if plan is None:
            nodes.append(node)
        else:
            construction = sample_matrix_plan(plan, rng)
            nodes.append(
                MatrixCausalNode(
                    parents=node.parents,
                    weights=construction.materialize(),
                    bias=node.bias,
                    noise_scale=node.noise_scale,
                    weight_construction=construction,
                )
            )
    fitted = fit_source_nodes(
        tuple(nodes),
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
        nodes=fitted,
        version="causal-v2",
        construction="source-shared-matrices-v1",
        source_prior=prior,
        source_selections=selections,
        rescale_preferences=scales,
        source_bindings=frozen_bindings,
        noise_units=noise_units,
        calibration_seed=seed,
        calibration_task_id=task_id,
        matrix_prior=matrix_prior,
        numerical_preferences=numerical,
        categorical_preferences=categorical,
        preference_bindings=matrix_bindings,
        matrix_activation_nodes=tuple(matrix_nodes),
    )
    return SourceMatrixCausalRecipe(**values)


SOURCE_MATRIX_PROFILES = {
    name.replace("-v1", "-matrices-v1"): (
        prior,
        CausalMatrixPrior(mode=prior.mode, categorical_policy=prior.policy),
    )
    for name, prior in SOURCE_ACTIVATION_PROFILES.items()
}
