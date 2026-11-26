"""Activation hyperpriors at the shared sampling/generation/replay seam."""

import json
from dataclasses import asdict, replace

from r0.modules.priors import (
    ActivatedCausalNode,
    CausalActivation,
    CausalActivationPrior,
    CausalGraphPrior,
    LinearRecipe,
    SampledCausalRecipe,
    explain_generation,
    generate_task,
    recipe_from_dict,
    sample_causal_recipe,
)


def test_homogeneous_activation_draw_is_frozen_and_rows_do_not_resample_it() -> None:
    prior = CausalActivationPrior(
        choices=(CausalActivation("tanh"), CausalActivation("sin")), mode="homogeneous"
    )
    graph = CausalGraphPrior(nodes=8)
    controls = LinearRecipe(
        rows=16, context_rows=8, features=2, signal_features=2, active_features=2
    )
    recipe = sample_causal_recipe(graph, task=controls, seed=28, task_id=3, activations=prior)
    assert isinstance(recipe, SampledCausalRecipe)
    assert all(isinstance(node, ActivatedCausalNode) for node in recipe.nodes)
    outputs = [
        node.output_activation for node in recipe.nodes if isinstance(node, ActivatedCausalNode)
    ]
    assert len(set(outputs)) == 1
    assert outputs[0] in prior.choices
    probabilities = recipe.activation_probabilities
    assert sorted(probabilities) == [0, 1]
    assert prior.choices[probabilities.index(1)] == outputs[0]
    more = sample_causal_recipe(
        graph, task=replace(controls, rows=32), seed=28, task_id=3, activations=prior
    )
    assert more.nodes == recipe.nodes
    assert isinstance(more, SampledCausalRecipe)
    assert more.activation_probabilities == probabilities
    data = json.loads(json.dumps(asdict(recipe)))
    restored = recipe_from_dict(data)
    assert restored == recipe
    task = generate_task(recipe, seed=28, task_id=3)
    assert generate_task(restored, seed=28, task_id=3).episode() == task.episode()
    card = json.loads(task.mechanism_json)
    assert card["activation_sampling"]["prior"] == json.loads(json.dumps(asdict(prior)))
    assert card["activation_sampling"]["probabilities"] == list(probabilities)
    assert explain_generation(task, [0, 15])["target_max_error"] < 1e-12
