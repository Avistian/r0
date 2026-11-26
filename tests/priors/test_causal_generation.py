"""Causal tasks use the shared interface and independently worked equations."""

from dataclasses import asdict

import torch

from r0.modules.priors import (
    CausalNode,
    CausalRecipe,
    LinearRecipe,
    explain_generation,
    generate_task,
    recipe_from_dict,
)


def test_authored_causal_graph_generates_the_worked_target() -> None:
    # docs/research/0028-worked-mechanisms.md: A=1, B=2*A+.5,
    # C=-A+3*B-1=5.5. This literal oracle is independent of the evaluator.
    controls = LinearRecipe(
        rows=16, context_rows=8, features=2, signal_features=2, active_features=2
    )
    recipe = recipe_from_dict(
        {
            "version": "causal-v1",
            "task": asdict(controls),
            "nodes": [
                {"parents": [], "weights": [], "bias": [1.0], "noise_scale": [0.0]},
                {
                    "parents": [0],
                    "weights": [[2.0]],
                    "bias": [0.5],
                    "noise_scale": [0.0],
                },
                {
                    "parents": [0, 1],
                    "weights": [[-1.0], [3.0]],
                    "bias": [-1.0],
                    "noise_scale": [0.0],
                },
            ],
            "observed": [[0, 0], [1, 0]],
            "target": [[2, 0]],
            "normalize": False,
        }
    )
    task = generate_task(recipe, seed=28, task_id=0)
    assert torch.equal(task.inputs, torch.tensor([[1.0, 2.5]] * 16, dtype=torch.float64))
    assert torch.equal(task.targets, torch.full((16,), 5.5, dtype=torch.float64))
    assert torch.equal(task.scores[:, 0], task.targets)
    assert task.episode().context[0].values == (1.0, 2.5)


def test_causal_trace_exposes_innovations_for_an_independent_equation_check() -> None:
    controls = LinearRecipe(
        rows=16, context_rows=8, features=1, signal_features=1, active_features=1
    )
    recipe = CausalRecipe(
        task=controls,
        nodes=(
            CausalNode(noise_scale=(1.0,)),
            CausalNode(parents=(0,), weights=((2.0,),), bias=(0.5,), noise_scale=(0.1,)),
        ),
        observed=((0, 0),),
        target=((1, 0),),
    )
    task = generate_task(recipe, seed=28, task_id=1)
    before = task.mechanism_json
    trace = explain_generation(task, [0, 5, 15])
    nodes = {node["id"]: node for node in trace["nodes"]}
    for j, row in enumerate((0, 5, 15)):
        x = nodes["causal-0"]["values"][j][0]
        innovation = nodes["innovation-1"]["values"][j][0]
        # Scalar structural equation, independent of the matrix evaluator.
        expected = 2 * x + 0.5 + innovation
        assert abs(float(task.targets[row]) - expected) < 1e-12
        assert nodes["causal-1"]["values"][j][0] == float(task.targets[row])
        assert x == float(task.inputs[row, 0])
    assert nodes["causal-1"]["parents"] == ["causal-0", "innovation-1"]
    assert task.mechanism_json == before


def test_vector_channels_and_replay_keep_the_actual_observation_dependencies() -> None:
    import json

    controls = LinearRecipe(
        rows=16, context_rows=8, features=2, signal_features=2, active_features=2
    )
    recipe = CausalRecipe(
        task=controls,
        nodes=(
            CausalNode(bias=(1.0, -2.0), noise_scale=(0.0, 0.0)),
            CausalNode(
                parents=(0,),
                weights=((2.0, 1.0), (-1.0, 3.0)),
                bias=(0.5, -0.5),
                noise_scale=(0.0, 0.0),
            ),
            CausalNode(parents=(1,), weights=((1.0,), (2.0,)), noise_scale=(0.0,)),
        ),
        observed=((0, 0), (1, 1)),
        target=((2, 0),),
    )
    task = generate_task(recipe, seed=28, task_id=2)
    assert torch.equal(task.targets, torch.full((16,), -6.5, dtype=torch.float64))
    assert torch.equal(task.inputs, torch.tensor([[1.0, -5.5]] * 16, dtype=torch.float64))
    trace = explain_generation(task, [2])
    nodes = {node["id"]: node for node in trace["nodes"]}
    assert nodes["causal-1"]["values"] == [[4.5, -5.5]]
    assert nodes["inputs"]["parents"] == ["causal-0", "causal-1"]
    assert nodes["inputs"]["parameters"]["observed_channels"] == [[0, 0], [1, 1]]
    replay = generate_task(
        recipe_from_dict(json.loads(task.recipe_json)["recipe"]), seed=28, task_id=2
    )
    assert replay.episode() == task.episode()


def test_causal_calibration_does_not_depend_on_query_rows_or_partition() -> None:
    import json
    from dataclasses import replace

    controls = LinearRecipe(
        rows=16, context_rows=8, features=1, signal_features=1, active_features=1
    )
    recipe = CausalRecipe(
        task=controls,
        nodes=(
            CausalNode(),
            CausalNode(parents=(0,), weights=((3.0,),), bias=(2.0,), noise_scale=(0.2,)),
        ),
        observed=((0, 0),),
        target=((1, 0),),
        normalize=True,
    )
    short = generate_task(recipe, seed=28, task_id=3)
    long = generate_task(
        replace(recipe, task=replace(controls, rows=32, context_rows=10, split=9)),
        seed=28,
        task_id=3,
    )
    assert (
        json.loads(short.mechanism_json)["calibration"]
        == json.loads(long.mechanism_json)["calibration"]
    )
    torch.testing.assert_close(short.inputs, long.inputs[:16], rtol=0, atol=0)
    torch.testing.assert_close(short.scores, long.scores[:16], rtol=1e-12, atol=1e-12)
    graph = explain_generation(short, [0, 15])
    signal = next(node for node in graph["nodes"] if node["id"] == "signal")
    torch.testing.assert_close(
        torch.tensor(signal["values"], dtype=torch.float64), short.scores[[0, 15]]
    )


def test_constant_disconnected_target_is_retained_and_flagged() -> None:
    import json

    controls = LinearRecipe(
        rows=16, context_rows=8, features=1, signal_features=1, active_features=1
    )
    recipe = CausalRecipe(
        task=controls,
        nodes=(CausalNode(), CausalNode(bias=(7.0,), noise_scale=(0.0,))),
        observed=((0, 0),),
        target=((1, 0),),
        normalize=True,
    )
    task = generate_task(recipe, seed=28, task_id=4)
    assert torch.equal(task.targets, torch.full((16,), 7.0, dtype=torch.float64))
    card = json.loads(task.mechanism_json)
    assert card["active"] == []
    assert "zero_target_dispersion" in card["flags"]
    assert "disconnected_target" in card["flags"]
    assert card["calibration"]["constant_channels"] == [0]
    assert card["calibration"]["center"] == [0.0]
    assert card["calibration"]["scale"] == [1.0]
