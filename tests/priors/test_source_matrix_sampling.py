"""Joint matrix/node activation preferences survive generation and saved replay."""

import json
from dataclasses import asdict, replace

import pytest

from r0.modules import priors


def test_global_plain_selector_is_shared_by_matrix_and_node_stages() -> None:
    controls = priors.LinearRecipe(
        rows=16, context_rows=8, features=2, signal_features=2, active_features=2
    )
    base = priors.sample_causal_recipe(
        priors.CausalGraphPrior(family="forward", nodes=4, edge_probability=1.0),
        task=controls,
        seed=28,
        task_id=0,
        widths=(2, 1, 2, 1),
    )
    source = priors.SourceActivationPrior(selector="plain", mode="global", placement="both")
    matrices = priors.CausalMatrixPrior(
        mode="global", families=("transformed",), simple_families=("gaussian",)
    )
    recipe = priors.with_source_matrices(
        base, prior=source, matrix_prior=matrices, seed=28, task_id=0
    )
    assert isinstance(recipe, priors.SourceMatrixCausalRecipe)
    assert len(recipe.matrix_activation_nodes) == 3
    assert len(recipe.source_selections.selections) == 10  # Three matrices, seven node stages.
    chosen = recipe.source_selections.selections[0].activation
    assert all(s.activation == chosen for s in recipe.source_selections.selections)
    for i in recipe.matrix_activation_nodes:
        node = recipe.nodes[i]
        assert isinstance(node, priors.MatrixCausalNode)
        assert isinstance(node.weight_construction, priors.NormalizedWeights)
        assert isinstance(node.weight_construction.source, priors.TransformedWeights)
        assert node.weight_construction.source.activation == chosen
    longer = priors.with_source_matrices(
        replace(base, task=replace(controls, rows=32, context_rows=16)),
        prior=source,
        matrix_prior=matrices,
        seed=28,
        task_id=0,
    )
    assert longer.nodes == recipe.nodes
    encoded = json.loads(json.dumps(asdict(recipe)))
    assert priors.recipe_from_dict(encoded) == recipe
    task = priors.generate_task(recipe, seed=28, task_id=0)
    assert priors.explain_generation(task, [0, 7, 15])["target_max_error"] < 1e-12
    encoded["matrix_activation_nodes"].pop()
    with pytest.raises(ValueError, match="matrix.*activation|binding"):
        priors.recipe_from_dict(encoded)


@pytest.mark.parametrize("family", ["forward", "layered", "redirection"])
@pytest.mark.parametrize("selector", ["standard", "plain"])
@pytest.mark.parametrize("suffix", ["", "-refreshed", "-local", "-global"])
def test_joint_source_matrix_profiles_are_available_in_public_sampling(
    family: str, selector: str, suffix: str
) -> None:
    recipe = priors.sample_recipe(
        family,
        task=priors.LinearRecipe(rows=16, context_rows=8),
        seed=28,
        task_id=10,
        profile=f"causal-source-{selector}{suffix}-matrices-v1",
    )
    assert isinstance(recipe, priors.SourceMatrixCausalRecipe)
    assert priors.recipe_from_dict(json.loads(json.dumps(asdict(recipe)))) == recipe


@pytest.mark.parametrize(
    "family", ["gaussian", "importance", "factorized", "kernel", "transformed"]
)
@pytest.mark.parametrize("after", [0.0, 1.0])
def test_standard_matrix_stages_freeze_calibration_and_admit_combined_storage(
    family: str, after: float
) -> None:
    controls = priors.LinearRecipe(
        rows=16, context_rows=8, features=2, signal_features=2, active_features=2
    )
    base = priors.sample_causal_recipe(
        priors.CausalGraphPrior(family="forward", nodes=4, edge_probability=1.0),
        task=controls,
        seed=28,
        task_id=0,
        widths=(2, 1, 2, 1),
    )
    source = priors.SourceActivationPrior(
        placement="both", calibration_rows=32, noise_after_probability=after
    )
    matrices = priors.CausalMatrixPrior(families=(family,), simple_families=("kernel",))
    recipe = priors.with_source_matrices(
        base, prior=source, matrix_prior=matrices, seed=28, task_id=0
    )
    longer = priors.with_source_matrices(
        replace(base, task=replace(controls, rows=32, context_rows=16)),
        prior=source,
        matrix_prior=matrices,
        seed=28,
        task_id=0,
    )
    assert longer.nodes == recipe.nodes
    encoded = json.loads(json.dumps(asdict(recipe)))
    assert priors.recipe_from_dict(encoded) == recipe
    generated = priors.generate_task(recipe, seed=28, task_id=0)
    assert priors.explain_generation(generated, [0, 7, 15])["target_max_error"] < 1e-12
    admission = json.loads(generated.mechanism_json)["resource_admission"]
    exact = replace(
        recipe.limits,
        parameters=admission["parameters"],
        working_bytes=admission["estimated_working_bytes"],
    )
    assert (
        priors.with_source_matrices(
            replace(base, limits=exact),
            prior=source,
            matrix_prior=matrices,
            seed=28,
            task_id=0,
        ).nodes
        == recipe.nodes
    )
    for key in ("parameters", "working_bytes"):
        with pytest.raises(ValueError, match="limit"):
            priors.with_source_matrices(
                replace(base, limits=replace(exact, **{key: getattr(exact, key) - 1})),
                prior=source,
                matrix_prior=matrices,
                seed=28,
                task_id=0,
            )
    encoded["nodes"][1]["output_activation"]["input_mean"][0] += 0.25
    with pytest.raises(ValueError, match="calibration"):
        priors.recipe_from_dict(encoded)
    encoded = json.loads(json.dumps(asdict(recipe)))
    encoded["preference_bindings"].pop()
    with pytest.raises(ValueError, match="binding"):
        priors.recipe_from_dict(encoded)
