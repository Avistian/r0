"""Shared sampling preferences stay bound to their generated causal matrices."""

import json
from dataclasses import asdict

import numpy as np
import pytest

from r0.modules import priors


@pytest.mark.parametrize(
    "family", ["gaussian", "importance", "factorized", "kernel", "transformed"]
)
def test_preferred_matrix_decisions_survive_generation_replay_and_binding_validation(
    family: str,
) -> None:
    controls = priors.LinearRecipe(
        rows=16, context_rows=8, features=2, active_features=2, signal_features=2
    )
    base = priors.sample_causal_recipe(
        priors.CausalGraphPrior(family="forward", nodes=4, edge_probability=1.0),
        task=controls,
        seed=28,
        task_id=1,
        widths=(1, 2, 2, 1),
        activations=priors.CausalActivationPrior(),
    )
    assert isinstance(base, priors.SampledCausalRecipe)
    prior = priors.CausalMatrixPrior(
        families=(family,), simple_families=("kernel",), point_laws=("gaussian-mixture",)
    )
    recipe = priors.sample_preferred_matrices(base, rng=np.random.default_rng(28), prior=prior)
    assert isinstance(recipe, priors.PreferredCausalRecipe)
    assert recipe.matrix_prior == prior
    assert recipe.categorical_preferences.draws
    assert recipe.preference_bindings
    assert recipe.sampling_parameter_count > 0
    encoded = json.loads(json.dumps(asdict(recipe)))
    restored = priors.recipe_from_dict(encoded)
    assert restored == recipe
    generated = priors.generate_task(recipe, seed=28, task_id=1)
    assert priors.generate_task(restored, seed=28, task_id=1).episode() == generated.episode()
    card = json.loads(generated.mechanism_json)
    assert (
        card["activation_sampling"]["categorical_preferences"] == encoded["categorical_preferences"]
    )
    without_preferences = recipe.limits.admit(
        recipe.nodes, rows=16, calibration_rows=0, features=2, outputs=1
    )["parameters"]
    assert (
        card["resource_admission"]["parameters"]
        == without_preferences + recipe.sampling_parameter_count
    )
    encoded["preference_bindings"].pop()
    with pytest.raises(ValueError, match="binding"):
        priors.recipe_from_dict(encoded)
