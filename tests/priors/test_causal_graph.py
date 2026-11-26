"""Topology contracts at the public sampler, using independently known graphs."""

import json
from dataclasses import asdict, replace

import torch

from r0.modules.priors import (
    CausalGraphPrior,
    LinearRecipe,
    explain_generation,
    generate_task,
    recipe_from_dict,
    sample_causal_graph,
    sample_causal_recipe,
)


def test_forward_graph_keeps_disconnected_components_and_extreme_probabilities() -> None:
    prior = CausalGraphPrior(nodes=6, components=2, edge_probability=1.0)
    full = sample_causal_graph(prior, seed=28, task_id=0)
    # Two disjoint three-vertex complete DAGs in parent-before-child order.
    assert full.parents == ((), (0,), (0, 1), (), (3,), (3, 4))
    assert full.prior == prior
    empty = sample_causal_graph(replace(prior, edge_probability=0), seed=28, task_id=0)
    assert empty.parents == ((),) * 6
    sparse = replace(prior, nodes=18, components=1, edge_probability=0.3)
    draw = sample_causal_graph(sparse, seed=28, task_id=1)
    assert draw == sample_causal_graph(sparse, seed=28, task_id=1)
    assert draw.parents != sample_causal_graph(sparse, seed=28, task_id=2).parents


def test_layered_graph_respects_layer_boundaries_in_each_component() -> None:
    prior = CausalGraphPrior(
        family="layered", nodes=10, components=2, layer_widths=(2, 1, 2), edge_probability=1
    )
    graph = sample_causal_graph(prior, seed=28, task_id=3)
    assert graph.parents == ((), (), (0, 1), (2,), (2,), (), (), (5, 6), (7,), (7,))
    empty = sample_causal_graph(replace(prior, edge_probability=0), seed=28, task_id=3)
    assert empty.parents == ((),) * 10


def test_sampled_vector_graph_is_a_replayable_task_with_saved_topology_provenance() -> None:
    controls = LinearRecipe(
        rows=16, context_rows=8, features=2, signal_features=2, active_features=2
    )
    graph = CausalGraphPrior(nodes=6, components=2, edge_probability=1)
    sampled = sample_causal_recipe(
        graph, task=controls, seed=28, task_id=4, widths=(2, 1, 3, 1, 2, 2)
    )
    assert tuple(node.parents for node in sampled.nodes) == ((), (0,), (0, 1), (), (3,), (3, 4))
    assert tuple(len(node.bias) for node in sampled.nodes) == (2, 1, 3, 1, 2, 2)
    assert not set(sampled.observed) & set(sampled.target)
    task = generate_task(sampled, seed=28, task_id=4)
    replay = generate_task(
        recipe_from_dict(json.loads(json.dumps(asdict(sampled)))), seed=28, task_id=4
    )
    assert replay.episode() == task.episode()
    card = json.loads(task.mechanism_json)
    assert card["topology"]["prior"] == json.loads(json.dumps(asdict(graph)))
    assert card["topology"]["seed"] == 28
    assert card["construction"] == "gaussian-affine-v1"
    trace = explain_generation(task, [0, 15])
    by_id = {node["id"]: node for node in trace["nodes"]}
    for column, (node, channel) in enumerate(sampled.observed):
        assert by_id[f"causal-{node}"]["values"][0][channel] == float(task.inputs[0, column])
    more_rows = sample_causal_recipe(
        graph,
        task=replace(controls, rows=32, context_rows=10),
        seed=28,
        task_id=4,
        widths=(2, 1, 3, 1, 2, 2),
    )
    assert more_rows.nodes == sampled.nodes
    longer = generate_task(more_rows, seed=28, task_id=4)
    torch.testing.assert_close(task.inputs, longer.inputs[:16], rtol=0, atol=0)
