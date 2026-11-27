"""Target calibration stays independent of emitted rows and replays retained constants."""

import json
from dataclasses import asdict, replace

import torch

from r0.modules import priors


def test_target_calibration_is_fixed_across_emission_sizes_and_inspection() -> None:
    recipe = priors.CalibratedFunctionalCausalRecipe(
        priors.LinearRecipe(
            rows=16, context_rows=8, features=1, signal_features=1, active_features=1
        ),
        (
            priors.FunctionalCausalNode(
                (),
                priors.FrozenLinearFunction(((1.0,), (4.0,))),
                root=priors.JointRootBase("gaussian", 1),
            ),
        ),
        ((0, 0),),
        ((0, 1),),
        calibration_rows=64,
    )
    decoded = priors.recipe_from_dict(json.loads(json.dumps(asdict(recipe))))
    assert decoded == recipe
    small = priors.generate_task(decoded, seed=47, task_id=9)
    larger = priors.generate_task(
        replace(recipe, task=replace(recipe.task, rows=128)), seed=47, task_id=9
    )
    small_card, larger_card = json.loads(small.mechanism_json), json.loads(larger.mechanism_json)
    assert small_card["calibration"] == larger_card["calibration"]
    calibration = small_card["calibration"]
    assert calibration["rows"] == 64
    assert calibration["stream"] == "target-calibration-v1"
    assert len(calibration["sample_digest"]) == 64
    expected = (4 * small.inputs - calibration["center"][0]) / calibration["scale"][0]
    torch.testing.assert_close(small.scores, expected)
    assert not torch.allclose(small.scores.mean(0), torch.zeros(1, dtype=torch.float64))
    trace = priors.explain_generation(small, [15, 0, 15])
    signal = next(n for n in trace["nodes"] if n["id"] == "signal")
    assert signal["parameters"]["calibration"] == calibration
    assert trace["signal_max_error"] == trace["target_max_error"] == 0


def test_normalization_requires_population_moments_and_keeps_constant_targets() -> None:
    import pytest

    controls = priors.LinearRecipe(
        rows=16, context_rows=8, features=1, signal_features=1, active_features=1
    )
    root = priors.FunctionalCausalNode(
        (),
        priors.FrozenLinearFunction(((1.0,), (2.0,))),
        root=priors.JointRootBase("uniform", 1),
        noise_scale=(0.0, 1.0),
        exogenous=priors.ExogenousDistribution(priors.InputDistribution("student_t", df=1.5)),
    )
    with pytest.raises(ValueError, match="finite second moments"):
        priors.CalibratedFunctionalCausalRecipe(controls, (root,), ((0, 0),), ((0, 1),))
    unnormalized = priors.FunctionalCausalRecipe(controls, (root,), ((0, 0),), ((0, 1),))
    assert torch.isfinite(priors.generate_task(unnormalized, seed=4, task_id=5).scores).all()
    constant = replace(
        root,
        function=priors.FrozenTreeFunction(1, ((0,),), ((0.0,),), (((1.0, 31.0), (2.0, 31.0)),)),
        noise_scale=(),
    )
    recipe = priors.CalibratedFunctionalCausalRecipe(controls, (constant,), ((0, 0),), ((0, 1),))
    task = priors.generate_task(recipe, seed=4, task_id=5)
    card = json.loads(task.mechanism_json)
    assert card["calibration"]["constant_channels"] == [0]
    assert card["calibration"]["center"] == [0.0]
    assert card["calibration"]["scale"] == [1.0]
    assert torch.equal(task.scores, torch.full((16, 1), 31.0, dtype=torch.float64))
    assert "zero_target_dispersion" in card["flags"]


def test_fixed_calibration_is_bound_to_graph_and_original_scores() -> None:
    import pytest

    recipe = priors.CalibratedFunctionalCausalRecipe(
        priors.LinearRecipe(
            rows=16, context_rows=8, features=1, signal_features=1, active_features=1
        ),
        (
            priors.FunctionalCausalNode(
                (),
                priors.FrozenLinearFunction(((1.0,), (2.0,))),
                root=priors.JointRootBase("gaussian", 1),
            ),
        ),
        ((0, 0),),
        ((0, 1),),
    )
    task = priors.generate_task(recipe, seed=1, task_id=2)
    card = json.loads(task.mechanism_json)
    card["calibration"]["prefix_fingerprint"] = "0" * 64
    with pytest.raises(ValueError, match="differs from its frozen graph"):
        priors.explain_generation(replace(task, mechanism_json=json.dumps(card)), [0, 1])
    card = json.loads(task.mechanism_json)
    card["calibration"]["center"][0] += 1
    with pytest.raises(ValueError, match="reconstructed signal differs"):
        priors.explain_generation(replace(task, mechanism_json=json.dumps(card)), [0, 1])
