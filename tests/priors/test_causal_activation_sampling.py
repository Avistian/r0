"""Activation hyperpriors at the shared sampling/generation/replay seam."""

import json
from dataclasses import asdict, replace

import pytest

from r0.modules.priors import (
    ActivatedCausalNode,
    ActivationParameterPrior,
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


def test_independent_and_shared_choices_preserve_their_distinct_probability_laws() -> None:
    controls = LinearRecipe(
        rows=16, context_rows=8, features=1, signal_features=1, active_features=1
    )
    graph = CausalGraphPrior(nodes=8, edge_probability=0)
    choices = (CausalActivation("tanh"), CausalActivation("sin"), CausalActivation("exp"))
    independent = CausalActivationPrior(choices=choices, weights=(1, 1, 0))
    shared = replace(independent, mode="dirichlet", concentration=2)
    probabilities: list[float] = []
    agreements = {"independent": 0, "dirichlet": 0}
    for task_id in range(512):
        for prior in (independent, shared):
            recipe = sample_causal_recipe(
                graph, task=controls, seed=59, task_id=task_id, activations=prior
            )
            assert isinstance(recipe, SampledCausalRecipe)
            assert recipe.activation_probabilities[2] == 0
            outputs = [
                n.output_activation.kind for n in recipe.nodes if isinstance(n, ActivatedCausalNode)
            ]
            assert "exp" not in outputs
            agreements[prior.mode] += outputs[0] == outputs[1]
            if prior.mode == "independent":
                assert recipe.activation_probabilities == (0.5, 0.5, 0)
            else:
                probabilities.append(recipe.activation_probabilities[0])
    # Dirichlet(1,1) is Uniform(0,1), mean 1/2, second moment 1/3.
    assert abs(sum(probabilities) / 512 - 0.5) < 0.05
    assert abs(sum(p * p for p in probabilities) / 512 - 1 / 3) < 0.05
    # Two independent fair choices agree with probability1/2; shared Uniform p gives2/3.
    assert abs(agreements["independent"] / 512 - 0.5) < 0.07
    assert abs(agreements["dirichlet"] / 512 - 2 / 3) < 0.07


def test_activation_parameter_distributions_freeze_values_and_homogeneous_parameters() -> None:
    prior = CausalActivationPrior(
        choices=(CausalActivation("signed_power"),),
        parameter_priors=(
            (
                ActivationParameterPrior("exponent", 0.1, 10, "log_uniform"),
                ActivationParameterPrior("output_shift", -2, 2),
            ),
        ),
        placement="both",
    )
    graph = CausalGraphPrior(nodes=16, edge_probability=1)
    controls = LinearRecipe(
        rows=16, context_rows=8, features=1, signal_features=1, active_features=1
    )
    recipe = sample_causal_recipe(graph, task=controls, seed=28, task_id=8, activations=prior)
    assert isinstance(recipe, SampledCausalRecipe)
    outputs = [
        node.output_activation for node in recipe.nodes if isinstance(node, ActivatedCausalNode)
    ]
    assert len(set(outputs)) > 1
    for activation in outputs:
        params = dict(activation.parameters)
        assert 0.1 <= params["exponent"] <= 10
        assert -2 <= params["output_shift"] <= 2
    assert recipe_from_dict(json.loads(json.dumps(asdict(recipe)))) == recipe
    homogeneous = sample_causal_recipe(
        graph, task=controls, seed=28, task_id=8, activations=replace(prior, mode="homogeneous")
    )
    assert (
        len({n.output_activation for n in homogeneous.nodes if isinstance(n, ActivatedCausalNode)})
        == 1
    )
    for node in homogeneous.nodes:
        assert isinstance(node, ActivatedCausalNode)
        if node.parents:
            assert node.input_activation == node.output_activation


@pytest.mark.parametrize(
    "corruption", ["probabilities", "zero_weight", "missing_prior", "noise_order"]
)
def test_replay_rejects_inconsistent_activation_sampling_provenance(corruption: str) -> None:
    controls = LinearRecipe(
        rows=16, context_rows=8, features=1, signal_features=1, active_features=1
    )
    prior = CausalActivationPrior(
        choices=(CausalActivation("tanh"), CausalActivation("sin")), noise_after_probability=0
    )
    recipe = sample_causal_recipe(
        CausalGraphPrior(nodes=4), task=controls, seed=4, task_id=2, activations=prior
    )
    wire = json.loads(json.dumps(asdict(recipe)))
    if corruption == "probabilities":
        wire["activation_probabilities"] = [0.9, 0.1]
    elif corruption == "zero_weight":
        wire["activation_prior"]["weights"] = [1, 0]
    elif corruption == "missing_prior":
        del wire["activation_prior"], wire["activation_probabilities"]
    else:
        wire["nodes"][0]["noise_position"] = "after_output"
    with pytest.raises(ValueError, match="sampling|probabilit|noise"):
        recipe_from_dict(wire)


def test_unrepresentable_preference_weights_fail_without_removing_choices() -> None:
    controls = LinearRecipe(
        rows=16, context_rows=8, features=1, signal_features=1, active_features=1
    )
    graph = CausalGraphPrior(nodes=4)
    prior = CausalActivationPrior(
        choices=(CausalActivation("tanh"), CausalActivation("sin")), weights=(1e-308, 1e308)
    )
    with pytest.raises(ValueError, match="representable|underflow"):
        sample_causal_recipe(graph, task=controls, seed=28, task_id=0, activations=prior)
    valid = sample_causal_recipe(
        graph, task=controls, seed=28, task_id=0, activations=replace(prior, weights=(1e308, 1e308))
    )
    assert isinstance(valid, SampledCausalRecipe)
    assert valid.activation_probabilities == (0.5, 0.5)


def test_homogeneous_sampling_record_requires_exact_one_hot_probability() -> None:
    controls = LinearRecipe(
        rows=16, context_rows=8, features=1, signal_features=1, active_features=1
    )
    recipe = sample_causal_recipe(
        CausalGraphPrior(nodes=4),
        task=controls,
        seed=1,
        task_id=2,
        activations=CausalActivationPrior(mode="homogeneous"),
    )
    wire = json.loads(json.dumps(asdict(recipe)))
    wire["activation_probabilities"] = [1 - 5e-13]
    with pytest.raises(ValueError, match="homogeneous.*one-hot"):
        recipe_from_dict(wire)
