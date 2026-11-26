"""Nonlinear structural equations through generation, replay and row inspection."""

import json
from dataclasses import asdict, replace

import torch

from r0.modules.priors import (
    ActivatedCausalNode,
    CausalActivation,
    CausalNode,
    CausalRecipe,
    LinearRecipe,
    explain_generation,
    generate_task,
    recipe_from_dict,
)


def test_nonlinear_noise_order_is_explicit_and_replayed_in_the_row_trace() -> None:
    controls = LinearRecipe(
        rows=16, context_rows=8, features=1, signal_features=1, active_features=1
    )
    node = ActivatedCausalNode(
        parents=(0,),
        weights=((2.0,),),
        bias=(1.0,),
        noise_scale=(0.5,),
        output_activation=CausalActivation("relu"),
    )
    before = CausalRecipe(
        task=controls,
        nodes=(CausalNode(bias=(-2.0,), noise_scale=(0.0,)), node),
        observed=((0, 0),),
        target=((1, 0),),
        version="causal-v2",
    )
    after = replace(before, nodes=(before.nodes[0], replace(node, noise_position="after_output")))
    left, right = [generate_task(r, seed=28, task_id=8) for r in (before, after)]
    traces = [explain_generation(t, list(range(16))) for t in (left, right)]
    nodes = [{n["id"]: n for n in trace["nodes"]} for trace in traces]
    assert nodes[0]["innovation-1"]["values"] == nodes[1]["innovation-1"]["values"]
    for row in range(16):
        epsilon = nodes[0]["innovation-1"]["values"][row][0]
        # Literal scalar equations: relu(-3 + epsilon) versus relu(-3) + epsilon.
        assert float(left.targets[row]) == max(0.0, -3.0 + epsilon)
        assert float(right.targets[row]) == epsilon
        assert nodes[0]["causal-1"]["values"][row][0] == float(left.targets[row])
        assert nodes[1]["causal-1"]["values"][row][0] == float(right.targets[row])
    assert not torch.equal(left.targets, right.targets)
    replay = generate_task(
        recipe_from_dict(json.loads(json.dumps(asdict(after)))), seed=28, task_id=8
    )
    assert replay.episode() == right.episode()
    assert "after_output" in nodes[1]["causal-1"]["parameters"]["noise_position"]
