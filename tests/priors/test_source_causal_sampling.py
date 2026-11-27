"""Source activation priors fit independent ancestral samples, then freeze for replay."""

import json
from dataclasses import asdict, replace

import pytest

from r0.modules import priors


@pytest.mark.parametrize("selector", ["standard", "plain"])
def test_source_activation_stages_bind_sampling_and_replay_independent_of_task_rows(
    selector: str,
) -> None:
    controls = priors.LinearRecipe(
        rows=16, context_rows=8, features=2, signal_features=2, active_features=2
    )
    graph = priors.CausalGraphPrior(family="forward", nodes=4, edge_probability=1.0)
    prior = priors.SourceActivationPrior(
        selector=selector, mode="local", placement="both", calibration_rows=32
    )
    base = priors.sample_causal_recipe(
        graph, task=controls, seed=28, task_id=0, widths=(2, 2, 2, 1)
    )
    recipe = priors.with_source_activations(base, prior=prior, seed=28, task_id=0)
    assert isinstance(recipe, priors.SourceCausalRecipe)
    assert len(recipe.source_selections.selections) == 7  # Four outputs, three inputs.
    assert len(recipe.source_bindings) == 7
    assert len(recipe.rescale_preferences.draws) == (7 if selector == "standard" else 0)
    longer = priors.with_source_activations(
        replace(base, task=replace(controls, rows=32, context_rows=16)),
        prior=prior,
        seed=28,
        task_id=0,
    )
    assert longer.nodes == recipe.nodes
    assert longer.source_selections == recipe.source_selections
    encoded = json.loads(json.dumps(asdict(recipe)))
    restored = priors.recipe_from_dict(encoded)
    assert restored == recipe
    task = priors.generate_task(restored, seed=28, task_id=0)
    assert priors.explain_generation(task, [0, 7, 15])["target_max_error"] < 1e-12
    admitted = recipe.limits.admit(
        recipe.nodes,
        rows=16,
        calibration_rows=32 if selector == "standard" else 0,
        features=2,
        outputs=1,
        extra_parameters=recipe.sampling_parameter_count,
    )
    assert json.loads(task.mechanism_json)["resource_admission"] == admitted
    encoded["source_bindings"].pop()
    with pytest.raises(ValueError, match="binding"):
        priors.recipe_from_dict(encoded)
    if selector == "standard":
        encoded = json.loads(json.dumps(asdict(recipe)))
        encoded["nodes"][0]["output_activation"]["input_mean"][0] += 0.25
        with pytest.raises(ValueError, match="calibration"):
            priors.recipe_from_dict(encoded)
